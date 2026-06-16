from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .schema import Chunk
from .utils import ensure_dir, load_json, read_jsonl, resolve_path, write_json


CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")
LATIN_NUM_RE = re.compile(r"[A-Za-z]+[A-Za-z0-9_\-]*|\d+(?:\.\d+)?%?")
STRUCTURED_RE = re.compile(
    r"第[一二三四五六七八九十百千万\d]+[编章节条款项]|20\d{2}|19\d{2}|\d+(?:\.\d+)?\s*(?:亿元|万元|元|倍|日|天|个月|年|个工作日|%)|《[^》]{2,80}》"
)
FINANCE_TERMS = [
    "营业收入", "归属于上市公司股东", "归母净利润", "净利润", "经营活动现金流", "现金流量净额", "研发投入",
    "研发费用", "分红", "现金分红", "回购", "资产负债率", "主体信用评级", "债项评级", "发行规模", "主承销商",
    "受托管理人", "违约", "回售", "赎回", "转股价格", "募集资金", "身故保险金", "现金价值", "账户价值",
    "保险责任", "责任免除", "等待期", "免赔额", "退保", "保单贷款", "受益所有人", "客户尽职调查",
    "信息披露", "股东大会", "特别决议", "普通决议", "独立董事", "定期报告", "行政处罚",
]
ENTITY_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9]{2,40}(?:股份有限公司|有限责任公司|集团有限公司|控股集团有限公司|证券股份有限公司|银行|保险|基金|科技|电子|股份|集团|公司|办法|准则|指引|条例|细则|通知|报告|说明书|债券|年报|保险|险))"
)
DOC_ID_ALIASES_RE = re.compile(r"(?:fc_)?text\d{2}|pack2_text\d{2}|annual_[a-z0-9_]+_report|csrc_\d{4}(?:_att\d+)?|strict_v3_\d+|\b\d{1,2}\b")


def normalize_text(text: str) -> str:
    return text.lower().replace("\u3000", " ")


def chinese_ngrams(token: str) -> list[str]:
    if len(token) <= 2:
        return [token]
    grams = [token[i : i + 2] for i in range(len(token) - 1)]
    grams.extend(token[i : i + 3] for i in range(len(token) - 2))
    if len(token) <= 8:
        grams.append(token)
    return grams


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    tokens: list[str] = []
    tokens.extend(m.group(0) for m in STRUCTURED_RE.finditer(text))
    tokens.extend(m.group(0) for m in ENTITY_RE.finditer(text))
    tokens.extend(term.lower() for term in FINANCE_TERMS if term.lower() in text)
    tokens.extend(m.group(0).lower() for m in DOC_ID_ALIASES_RE.finditer(text))
    tokens.extend(m.group(0) for m in LATIN_NUM_RE.finditer(text))
    for match in CHINESE_RE.finditer(text):
        tokens.extend(chinese_ngrams(match.group(0)))
    return [t for t in tokens if len(t.strip()) > 0]


def extract_structured_terms(text: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for match in STRUCTURED_RE.finditer(text):
        term = match.group(0).strip()
        if term and term not in seen:
            seen.add(term)
            terms.append(term)
    return terms


class LexicalIndex:
    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self.chunks: list[Chunk] = []
        self.doc_freq: dict[str, int] = {}
        self.term_freqs: list[Counter[str]] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_len = 0.0
        self.doc_to_chunk_indices: dict[str, list[int]] = defaultdict(list)
        self.doc_metadata: dict[str, dict[str, Any]] = {}
        self.doc_profiles: dict[str, Counter[str]] = {}

    def build_from_processed(self, chunks_dir: Path, metadata_path: Path | None = None) -> "LexicalIndex":
        chunks: list[Chunk] = []
        for path in sorted(chunks_dir.glob("*.jsonl")):
            for row in read_jsonl(path):
                chunks.append(Chunk.from_dict(row))
        self._build(chunks)
        if metadata_path and metadata_path.exists():
            manifest = load_json(metadata_path)
            for item in manifest.get("documents", []):
                self.doc_metadata[item["doc_id"]] = item
        return self

    def _build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.term_freqs = []
        df: Counter[str] = Counter()
        self.doc_lengths = []
        self.doc_to_chunk_indices = defaultdict(list)
        self.doc_profiles = defaultdict(Counter)

        for idx, chunk in enumerate(chunks):
            tokens = tokenize(" ".join([chunk.title, *chunk.keywords, chunk.text]))
            counts = Counter(tokens)
            self.term_freqs.append(counts)
            self.doc_lengths.append(sum(counts.values()))
            df.update(counts.keys())
            self.doc_to_chunk_indices[chunk.doc_id].append(idx)
            profile_tokens = tokenize(" ".join([chunk.doc_id, chunk.title, *chunk.keywords]))
            profile_tokens.extend(extract_structured_terms(chunk.text))
            self.doc_profiles[chunk.doc_id].update(profile_tokens)

        self.doc_freq = dict(df)
        self.avg_doc_len = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0
        self.logger.info("built lexical index: %s chunks, %s docs", len(chunks), len(self.doc_to_chunk_indices))

    def save(self, path: Path) -> None:
        ensure_dir(path.parent)
        payload = {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "doc_metadata": self.doc_metadata,
            "doc_profiles": {doc_id: dict(counter) for doc_id, counter in self.doc_profiles.items()},
        }
        write_json(path, payload)

    @classmethod
    def load(cls, path: Path, logger: logging.Logger | None = None) -> "LexicalIndex":
        payload = load_json(path)
        index = cls(logger=logger)
        index.doc_metadata = payload.get("doc_metadata", {})
        index._build([Chunk.from_dict(row) for row in payload.get("chunks", [])])
        if payload.get("doc_profiles"):
            index.doc_profiles = {doc_id: Counter(counter) for doc_id, counter in payload.get("doc_profiles", {}).items()}
        return index

    def search_docs(
        self,
        query: str,
        domain: str | None = None,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        query_structured = set(extract_structured_terms(query))
        chunk_hits = self.search_chunks(query=query, domain=domain, top_k=max(top_k * 10, 40))
        scores: dict[str, float] = defaultdict(float)
        titles: dict[str, str] = {}
        domains: dict[str, str] = {}
        for hit in chunk_hits:
            doc_id = hit["chunk"].doc_id
            scores[doc_id] += hit["score"]
            titles[doc_id] = hit["chunk"].title
            domains[doc_id] = hit["chunk"].domain
        for doc_id, profile in self.doc_profiles.items():
            meta = self.doc_metadata.get(doc_id, {})
            doc_domain = str(meta.get("domain", domains.get(doc_id, "")))
            if domain and doc_domain != domain:
                continue
            profile_score = self._doc_profile_score(doc_id, profile, query_tokens, query_structured, query)
            if profile_score > 0:
                scores[doc_id] += profile_score
                titles.setdefault(doc_id, str(meta.get("title", doc_id)))
        return [
            {"doc_id": doc_id, "score": score, "title": titles.get(doc_id, "")}
            for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        ]

    def _doc_profile_score(
        self,
        doc_id: str,
        profile: Counter[str],
        query_tokens: list[str],
        query_structured: set[str],
        query: str,
    ) -> float:
        score = 0.0
        query_norm = normalize_text(query)
        if doc_id.lower() in query_norm:
            score += 12.0
        aliases = self._doc_aliases(doc_id)
        if any(alias and alias in query_norm for alias in aliases):
            score += 8.0
        for term in query_structured:
            if profile.get(term, 0):
                score += 2.0
        for token in set(query_tokens):
            if len(token) >= 2 and profile.get(token, 0):
                score += min(1.5, 0.2 * profile[token])
        return score

    def _doc_aliases(self, doc_id: str) -> list[str]:
        aliases = [doc_id.lower()]
        if doc_id.startswith("text"):
            aliases.append(f"fc_{doc_id}".lower())
        if doc_id.startswith("pack2_text"):
            aliases.append(doc_id.replace("pack2_", ""))
        if doc_id.startswith("annual_"):
            parts = doc_id.split("_")
            if len(parts) >= 3:
                aliases.extend([parts[1], parts[2]])
        return aliases

    def search_chunks(
        self,
        query: str,
        domain: str | None = None,
        doc_ids: set[str] | None = None,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        if not self.chunks:
            return []
        query_tokens = tokenize(query)
        query_counts = Counter(query_tokens)
        structured = set(extract_structured_terms(query))
        results: list[dict[str, Any]] = []
        allowed_indices: range | list[int]
        if doc_ids:
            allowed = []
            for doc_id in doc_ids:
                allowed.extend(self.doc_to_chunk_indices.get(doc_id, []))
            allowed_indices = allowed
        else:
            allowed_indices = range(len(self.chunks))

        for idx in allowed_indices:
            chunk = self.chunks[idx]
            if domain and chunk.domain != domain:
                continue
            score = self._bm25(idx, query_counts)
            matched_terms = self._matched_terms(chunk, query_tokens, structured)
            score += self._rule_boost(chunk, query, matched_terms, structured, doc_ids)
            if score <= 0:
                continue
            results.append({"chunk": chunk, "score": score, "matched_terms": matched_terms})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _bm25(self, idx: int, query_counts: Counter[str], k1: float = 1.5, b: float = 0.75) -> float:
        score = 0.0
        tf = self.term_freqs[idx]
        dl = self.doc_lengths[idx] or 1
        n = len(self.chunks)
        for term, qf in query_counts.items():
            f = tf.get(term, 0)
            if f <= 0:
                continue
            df = self.doc_freq.get(term, 0)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            denom = f + k1 * (1 - b + b * dl / (self.avg_doc_len or 1))
            score += idf * (f * (k1 + 1) / denom) * min(qf, 3)
        return score

    def _matched_terms(self, chunk: Chunk, query_tokens: list[str], structured: set[str]) -> list[str]:
        text = normalize_text(" ".join([chunk.title, chunk.text]))
        matched: list[str] = []
        seen: set[str] = set()
        for term in list(structured) + query_tokens:
            if len(term) < 2 or term in seen:
                continue
            if normalize_text(term) in text:
                seen.add(term)
                matched.append(term)
            if len(matched) >= 30:
                break
        return matched

    def _rule_boost(
        self,
        chunk: Chunk,
        query: str,
        matched_terms: list[str],
        structured: set[str],
        doc_ids: set[str] | None,
    ) -> float:
        score = 0.0
        query_norm = normalize_text(query)
        if chunk.doc_id in query_norm:
            score += 2.0
        for alias in self._doc_aliases(chunk.doc_id):
            if alias and alias in query_norm:
                score += 1.5
        if chunk.title and normalize_text(chunk.title) in query_norm:
            score += 1.5
        if doc_ids and chunk.doc_id in doc_ids:
            score += 1.0
        if structured:
            score += min(4.0, 0.8 * sum(1 for term in structured if term in matched_terms))
        score += min(2.0, 0.08 * len(matched_terms))
        return score


def build_and_save_index(config: dict[str, Any], logger: logging.Logger | None = None) -> Path:
    chunks_dir = resolve_path(config["chunks_dir"])
    metadata_path = resolve_path(config["metadata_dir"]) / "documents.json"
    index_dir = ensure_dir(resolve_path(config["index_dir"]))
    index_path = index_dir / "lexical_index.json"
    index = LexicalIndex(logger=logger).build_from_processed(chunks_dir, metadata_path)
    index.save(index_path)
    return index_path
