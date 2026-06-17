from __future__ import annotations

import logging
import re
from typing import Any

from .schema import EvidenceBundle, EvidenceItem, Question


IMPORTANT_RE = re.compile(
    r"(第[一二三四五六七八九十百千万\d]+[编章节条款项]|20\d{2}|19\d{2}|\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s*(?:亿元|万元|元|倍|日|天|个月|年|个工作日)|必须|应当|不得|无需|除外|例外|责任|免赔|现金价值|账户价值|净利润|营业收入|现金流|研发|评级|发行|违约|回售|赎回|转股|分红|回购|特别决议|普通决议)"
)
NEGATION_RE = re.compile(r"(不|未|无|不得|无需|不能|除外|免除|不承担|不适用|错误)")


class MemoryManager:
    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None):
        self.max_context_chars = int(config.get("max_context_chars", 16000))
        self.option_coverage_min = int(config.get("option_coverage_min", 1))
        self.enable_qwen_compression = bool(config.get("enable_qwen_compression", False))
        self.max_compressed_chars_per_chunk = int(config.get("max_compressed_chars_per_chunk", 1500))
        self.table_line_neighbor_window = int(config.get("table_line_neighbor_window", 1))
        self.logger = logger or logging.getLogger(__name__)

    def compact(self, question: Question, bundle: EvidenceBundle) -> tuple[str, list[EvidenceItem]]:
        selected: list[EvidenceItem] = self._option_coverage_items(question, bundle)
        total = 0
        seen_text: set[str] = set()
        seen_chunks: set[str] = set()
        formatted_selected: list[EvidenceItem] = []

        for item in selected + bundle.items:
            if item.chunk_id in seen_chunks:
                continue
            text = self._compress_text(question, item.text, item.option)
            if len(text) < 30:
                continue
            key = self._dedup_key(text)
            if key in seen_text:
                continue
            else:
                seen_text.add(key)
            seen_chunks.add(item.chunk_id)
            new_item = self._clone_with_text(item, text)
            block_len = len(self._format_item(new_item))
            if formatted_selected and total + block_len > self.max_context_chars:
                continue
            formatted_selected.append(new_item)
            total += block_len

        context = self._build_context(question, formatted_selected)
        return context[: self.max_context_chars], formatted_selected

    def _option_coverage_items(self, question: Question, bundle: EvidenceBundle) -> list[EvidenceItem]:
        selected: list[EvidenceItem] = []
        seen: set[str] = set()
        for option in sorted(question.options):
            count = 0
            for item in bundle.items:
                if item.option == option and item.chunk_id not in seen:
                    selected.append(item)
                    seen.add(item.chunk_id)
                    count += 1
                    if count >= self.option_coverage_min:
                        break
        return selected

    def _compress_text(self, question: Question, text: str, option: str | None = None) -> str:
        sentences = self._split_sentences(text)
        if len(text) <= self.max_compressed_chars_per_chunk:
            return text
        q_terms = self._query_terms(question)
        option_terms = self._option_terms(question, option)
        line_context = self._compress_by_lines(text, q_terms, option_terms)
        if line_context:
            return line_context
        ranked = sorted(
            ((self._sentence_score(sentence, q_terms, option_terms), idx, sentence) for idx, sentence in enumerate(sentences)),
            key=lambda x: (-x[0], x[1]),
        )
        kept_ranked = [sentence for score, _, sentence in ranked if score > 0]
        if not kept_ranked:
            kept_ranked = sentences[:4]
        kept: list[str] = []
        total = 0
        for sentence in kept_ranked:
            kept.append(sentence)
            total += len(sentence)
            if total >= self.max_compressed_chars_per_chunk:
                break
        return "\n".join(kept).strip()[: self.max_compressed_chars_per_chunk]

    def _sentence_score(self, sentence: str, q_terms: list[str], option_terms: list[str]) -> float:
        score = 0.0
        if IMPORTANT_RE.search(sentence):
            score += 3.0
        if NEGATION_RE.search(sentence):
            score += 1.2
        score += min(3.0, 0.4 * sum(1 for term in q_terms if term and term in sentence))
        score += min(4.0, 0.8 * sum(1 for term in option_terms if term and term in sentence))
        if len(sentence) > 260:
            score -= 0.5
        return score

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[。！？；;])", text)
        sentences = [part.strip() for part in parts if len(part.strip()) >= 8]
        if not sentences:
            sentences = [line.strip() for line in text.splitlines() if len(line.strip()) >= 8]
        return sentences

    def _compress_by_lines(self, text: str, q_terms: list[str], option_terms: list[str]) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 4:
            return ""
        scored = []
        for idx, line in enumerate(lines):
            score = self._sentence_score(line, q_terms, option_terms)
            if "|" in line or "\t" in line or re.search(r"\s{2,}", line):
                score += 0.8
            if re.search(r"项目|指标|本期|上期|202[0-9]|金额|比例|单位", line):
                score += 0.8
            scored.append((score, idx, line))
        hit_indices = {idx for score, idx, _ in scored if score > 0}
        if not hit_indices:
            return ""
        keep_indices: set[int] = set()
        for idx in hit_indices:
            for neighbor in range(idx - self.table_line_neighbor_window, idx + self.table_line_neighbor_window + 1):
                if 0 <= neighbor < len(lines):
                    keep_indices.add(neighbor)
        for idx, line in enumerate(lines[:6]):
            if re.search(r"项目|指标|单位|20\d{2}|本期|上期", line):
                keep_indices.add(idx)
        kept: list[str] = []
        total = 0
        for idx in sorted(keep_indices):
            line = lines[idx]
            kept.append(line)
            total += len(line) + 1
            if total >= self.max_compressed_chars_per_chunk:
                break
        return "\n".join(kept).strip()[: self.max_compressed_chars_per_chunk]

    def _query_terms(self, question: Question) -> list[str]:
        raw = [question.question, question.type, *question.options.values()]
        terms: set[str] = set()
        for text in raw:
            for token in re.findall(r"《[^》]+》|20\d{2}|\d+(?:\.\d+)?%|[\u4e00-\u9fff]{2,6}", text):
                if len(token) >= 2:
                    terms.add(token)
        return list(terms)[:80]

    def _option_terms(self, question: Question, option: str | None) -> list[str]:
        if not option or option not in question.options:
            return []
        text = question.options[option]
        return re.findall(r"《[^》]+》|20\d{2}|\d+(?:\.\d+)?%|[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_\-]{1,30}", text)[:60]

    def _dedup_key(self, text: str) -> str:
        return re.sub(r"\s+", "", text)[:220]

    def _clone_with_text(self, item: EvidenceItem, text: str) -> EvidenceItem:
        return EvidenceItem(
            evidence_id=item.evidence_id,
            chunk_id=item.chunk_id,
            doc_id=item.doc_id,
            domain=item.domain,
            title=item.title,
            page_start=item.page_start,
            page_end=item.page_end,
            section_path=item.section_path,
            text=text,
            score=item.score,
            matched_terms=item.matched_terms,
            option=item.option,
        )

    def _build_context(self, question: Question, items: list[EvidenceItem]) -> str:
        option_hints: list[str] = []
        by_option: dict[str, list[str]] = {}
        for item in items:
            if item.option:
                by_option.setdefault(item.option, []).append(item.evidence_id)
        for option in sorted(question.options):
            ids = ", ".join(by_option.get(option, [])[:5]) or "无专属证据，参考全局证据"
            option_hints.append(f"选项{option}证据提示：{ids}")
        evidence_text = "\n\n".join(self._format_item(item) for item in items)
        return "\n".join(option_hints) + "\n\n" + evidence_text

    def _format_item(self, item: EvidenceItem) -> str:
        section = " > ".join(item.section_path)
        page = f"{item.page_start}" if item.page_start == item.page_end else f"{item.page_start}-{item.page_end}"
        header = f"[{item.evidence_id}] doc_id={item.doc_id} title={item.title} page={page}"
        if section:
            header += f" section={section}"
        return f"{header}\n{item.text}"
