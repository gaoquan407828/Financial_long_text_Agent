from __future__ import annotations

import logging
import re
from collections import OrderedDict
from typing import Any

from .lexical_index import LexicalIndex, extract_structured_terms
from .schema import EvidenceBundle, EvidenceItem, Question


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
        for doc_id in question.doc_ids:
            queries.append(doc_id)
        queries.extend(self._doc_id_hints(question))
        return queries[:12]

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
