from __future__ import annotations

import logging
import re
from collections import OrderedDict
from typing import Any

from .lexical_index import LexicalIndex, extract_structured_terms, normalize_text
from .schema import EvidenceBundle, EvidenceItem, Question


METRIC_TERMS = [
    "营业收入", "营业总收入", "归母净利润", "归属于上市公司股东的净利润", "经营活动产生的现金流量净额",
    "经营活动现金流量净额", "经营现金流", "研发投入", "研发费用", "现金分红", "分红金额", "回购",
    "资产负债率", "发行规模", "发行金额", "注册金额", "主体信用评级", "债项信用评级", "主承销商",
    "受托管理人", "初始转股价格", "转股价格", "回售", "赎回", "违约", "补偿", "兑付日",
]
STRONG_WORDS_RE = re.compile(
    r"(必须|应当|不得|可以|无需|不能|不|未|无|除外|免除|错误|正确|超过|低于|高于|下降|增长|减少|增加|"
    r"AAA|AA\+|AA-|A\+|A-|10\s*股|每股|每\s*10\s*股)"
)
NUMBER_RE = re.compile(r"20\d{2}|19\d{2}|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|亿元|万元|元|倍|日|天|个月|年)?")


class Retriever:
    def __init__(self, index: LexicalIndex, config: dict[str, Any], logger: logging.Logger | None = None):
        self.index = index
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.top_k_docs = int(config.get("retrieve_top_k_docs", 8))
        self.stage1_top_k_docs = int(config.get("retrieve_stage1_top_k_docs", max(self.top_k_docs * 3, 16)))
        self.top_k_chunks = int(config.get("retrieve_top_k_chunks", 24))
        self.per_option_top_k = int(config.get("per_option_top_k", 5))
        self.per_doc_probe_top_k = int(config.get("per_doc_probe_top_k", 2))
        self.max_evidence_chars = int(config.get("max_evidence_chars_per_chunk", 1200))
        self.neighbor_window = int(config.get("neighbor_window", 0))
        self.neighbor_max_seed_items = int(config.get("neighbor_max_seed_items", 10))
        self.rerank_weight = float(config.get("evidence_rerank_weight", 1.0))
        self.numeric_match_boost = float(config.get("numeric_match_boost", 2.0))
        self.metric_match_boost = float(config.get("metric_match_boost", 1.4))
        self.option_match_boost = float(config.get("option_match_boost", 1.2))
        self.chunk_id_to_index = {chunk.chunk_id: idx for idx, chunk in enumerate(self.index.chunks)}

    def retrieve(self, question: Question) -> EvidenceBundle:
        candidate_doc_ids = set(question.doc_ids or [])
        query_all = self._question_query(question)
        if not candidate_doc_ids:
            doc_hits = self._retrieve_candidate_docs(question)
            candidate_doc_ids = {hit["doc_id"] for hit in doc_hits[: self.top_k_docs]}
            self.logger.info("%s candidate docs: %s", question.qid, sorted(candidate_doc_ids))

        merged: OrderedDict[str, EvidenceItem] = OrderedDict()
        global_hits = self.index.search_chunks(
            query=query_all,
            domain=question.domain,
            doc_ids=candidate_doc_ids or None,
            top_k=self.top_k_chunks,
        )
        self._add_hits(merged, global_hits, option=None)

        for option_key, option_text in question.options.items():
            option_query = f"{question.question}\n选项{option_key}: {option_text}"
            option_hits = self.index.search_chunks(
                query=option_query,
                domain=question.domain,
                doc_ids=candidate_doc_ids or None,
                top_k=self.per_option_top_k,
            )
            self._add_hits(merged, option_hits, option=option_key)

        exact_queries = self._exact_queries(question)
        for exact_query in exact_queries:
            exact_hits = self.index.search_chunks(
                query=exact_query,
                domain=question.domain,
                doc_ids=candidate_doc_ids or None,
                top_k=max(3, self.per_option_top_k),
            )
            self._add_hits(merged, exact_hits, option=None)

        if not question.doc_ids and candidate_doc_ids:
            for doc_id in candidate_doc_ids:
                probe_hits = self.index.search_chunks(
                    query=query_all,
                    domain=question.domain,
                    doc_ids={doc_id},
                    top_k=self.per_doc_probe_top_k,
                )
                self._add_hits(merged, probe_hits, option=None)

        self._add_neighbor_chunks(merged, question)
        items = list(merged.values())
        self._rerank_items(question, items)
        items.sort(key=lambda item: item.score, reverse=True)
        for idx, item in enumerate(items, start=1):
            item.evidence_id = f"E{idx}"
        used_doc_ids = sorted({item.doc_id for item in items})
        return EvidenceBundle(question_id=question.qid, items=items, used_doc_ids=used_doc_ids)

    def _retrieve_candidate_docs(self, question: Question) -> list[dict[str, Any]]:
        queries = [self._question_query(question), question.question, question.type]
        queries.extend(question.options.values())
        queries.extend(f"{question.question}\n{key}. {value}" for key, value in sorted(question.options.items()))
        queries.extend(self._exact_queries(question))
        scores: dict[str, dict[str, Any]] = {}
        for query in queries:
            if not query.strip():
                continue
            hits = self.index.search_docs(query, domain=question.domain, top_k=self.stage1_top_k_docs)
            for hit in hits:
                doc_id = hit["doc_id"]
                current = scores.setdefault(doc_id, {"doc_id": doc_id, "score": 0.0, "title": hit.get("title", "")})
                current["score"] += float(hit["score"])
                if not current.get("title"):
                    current["title"] = hit.get("title", "")

        for doc_id in self._doc_id_hints(question):
            if doc_id in self.index.doc_to_chunk_indices:
                current = scores.setdefault(doc_id, {"doc_id": doc_id, "score": 0.0, "title": doc_id})
                current["score"] += 20.0

        return sorted(scores.values(), key=lambda x: x["score"], reverse=True)

    def _add_hits(
        self,
        merged: OrderedDict[str, EvidenceItem],
        hits: list[dict[str, Any]],
        option: str | None,
    ) -> None:
        for hit in hits:
            chunk = hit["chunk"]
            text = chunk.text[: self.max_evidence_chars].strip()
            if not text:
                continue
            existing = merged.get(chunk.chunk_id)
            if existing:
                existing.score = max(existing.score, float(hit["score"]))
                if option and existing.option is None:
                    existing.option = option
                existing.matched_terms = sorted(set(existing.matched_terms) | set(hit.get("matched_terms", [])))
                continue
            merged[chunk.chunk_id] = EvidenceItem(
                evidence_id="",
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                domain=chunk.domain,
                title=chunk.title,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=chunk.section_path,
                text=text,
                score=float(hit["score"]),
                matched_terms=hit.get("matched_terms", []),
                option=option,
            )

    def _add_neighbor_chunks(self, merged: OrderedDict[str, EvidenceItem], question: Question) -> None:
        if self.neighbor_window <= 0 or not merged:
            return
        seeds = sorted(merged.values(), key=lambda item: item.score, reverse=True)[: self.neighbor_max_seed_items]
        for seed in seeds:
            seed_idx = self.chunk_id_to_index.get(seed.chunk_id)
            if seed_idx is None:
                continue
            for neighbor_idx in self._neighbor_indices(seed_idx, seed.doc_id):
                chunk = self.index.chunks[neighbor_idx]
                if question.domain and chunk.domain != question.domain:
                    continue
                if chunk.chunk_id in merged:
                    continue
                text = chunk.text[: self.max_evidence_chars].strip()
                if not text:
                    continue
                merged[chunk.chunk_id] = EvidenceItem(
                    evidence_id="",
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    domain=chunk.domain,
                    title=chunk.title,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_path=chunk.section_path,
                    text=text,
                    score=max(seed.score * 0.72, 0.01),
                    matched_terms=["neighbor_context"],
                    option=seed.option,
                )

    def _neighbor_indices(self, seed_idx: int, doc_id: str) -> list[int]:
        indices = self.index.doc_to_chunk_indices.get(doc_id, [])
        try:
            position = indices.index(seed_idx)
        except ValueError:
            return []
        start = max(0, position - self.neighbor_window)
        end = min(len(indices), position + self.neighbor_window + 1)
        return [indices[pos] for pos in range(start, end) if indices[pos] != seed_idx]

    def _question_query(self, question: Question) -> str:
        options = "\n".join(f"{k}. {v}" for k, v in sorted(question.options.items()))
        doc_hint = " ".join(question.doc_ids or [])
        return f"{question.question}\n{options}\n{question.type}\n{doc_hint}"

    def _exact_queries(self, question: Question) -> list[str]:
        text = self._question_query(question)
        terms = extract_structured_terms(text)
        queries: list[str] = []
        for term in terms:
            queries.append(f"{term} {question.domain}")
        for metric in self._metric_terms(text):
            queries.append(f"{metric} {' '.join(self._years(text))} {' '.join(question.doc_ids)}")
        for option_key, option_text in sorted(question.options.items()):
            option_metrics = self._metric_terms(option_text)
            option_numbers = self._numbers(option_text)
            strong_words = self._strong_words(option_text)
            if option_metrics or option_numbers or strong_words:
                queries.append(" ".join([question.question, option_key, *option_metrics, *option_numbers, *strong_words, *question.doc_ids]))
        for doc_id in question.doc_ids:
            queries.append(doc_id)
        queries.extend(self._doc_id_hints(question))
        return list(dict.fromkeys(q for q in queries if q.strip()))[:30]

    def _doc_id_hints(self, question: Question) -> list[str]:
        text = self._question_query(question)
        hints = re.findall(r"(?:fc_)?text\d{2}|pack2_text\d{2}|annual_[a-z0-9_]+_report|csrc_\d{4}(?:_att\d+)?|strict_v3_\d+", text, flags=re.I)
        normalized: list[str] = []
        for hint in hints:
            h = hint.lower()
            if h.startswith("fc_text"):
                h = h.replace("fc_", "", 1)
            normalized.append(h)
        return list(dict.fromkeys(normalized))

    def _rerank_items(self, question: Question, items: list[EvidenceItem]) -> None:
        if self.rerank_weight <= 0:
            return
        question_text = self._question_query(question)
        q_terms = self._important_terms(question_text)
        q_numbers = self._numbers(question_text)
        q_metrics = self._metric_terms(question_text)
        q_strong = self._strong_words(question_text)
        option_terms = {key: self._important_terms(value) for key, value in question.options.items()}
        option_numbers = {key: self._numbers(value) for key, value in question.options.items()}
        option_metrics = {key: self._metric_terms(value) for key, value in question.options.items()}

        for item in items:
            text = normalize_text(" ".join([item.title, *item.section_path, item.text]))
            bonus = 0.0
            bonus += min(5.0, 0.35 * self._count_contains(text, q_terms))
            bonus += self.numeric_match_boost * self._count_contains(text, q_numbers)
            bonus += self.metric_match_boost * self._count_contains(text, q_metrics)
            bonus += 0.8 * self._count_contains(text, q_strong)
            if item.option and item.option in question.options:
                bonus += self.option_match_boost * self._count_contains(text, option_terms[item.option])
                bonus += self.numeric_match_boost * self._count_contains(text, option_numbers[item.option])
                bonus += self.metric_match_boost * self._count_contains(text, option_metrics[item.option])
            if item.doc_id in question.doc_ids:
                bonus += 1.0
            if "neighbor_context" in item.matched_terms and bonus < 1.0:
                bonus -= 0.4
            item.score += bonus * self.rerank_weight

    def _important_terms(self, text: str) -> list[str]:
        terms: list[str] = []
        terms.extend(self._metric_terms(text))
        terms.extend(self._numbers(text))
        terms.extend(self._strong_words(text))
        for token in re.findall(r"《[^》]+》|[\u4e00-\u9fffA-Za-z0-9]{2,18}", text):
            token = token.strip()
            if len(token) >= 2 and token not in {"下列", "关于", "根据", "提供", "文档", "选项", "正确", "错误"}:
                terms.append(token)
        return list(dict.fromkeys(terms))[:80]

    def _metric_terms(self, text: str) -> list[str]:
        return [term for term in METRIC_TERMS if term.lower() in text.lower()]

    def _numbers(self, text: str) -> list[str]:
        return [re.sub(r"\s+", "", m.group(0)) for m in NUMBER_RE.finditer(text)]

    def _years(self, text: str) -> list[str]:
        return list(dict.fromkeys(re.findall(r"20\d{2}|19\d{2}", text)))

    def _strong_words(self, text: str) -> list[str]:
        return list(dict.fromkeys(m.group(0) for m in STRONG_WORDS_RE.finditer(text)))

    def _count_contains(self, text: str, terms: list[str]) -> int:
        count = 0
        normalized_text = re.sub(r"\s+", "", text.lower())
        for term in terms:
            normalized_term = re.sub(r"\s+", "", term.lower())
            if normalized_term and normalized_term in normalized_text:
                count += 1
        return count
