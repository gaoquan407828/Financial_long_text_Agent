from __future__ import annotations

import logging
import re
from typing import Any

from .schema import Chunk, Document
from .utils import stable_id


SECTION_RE = re.compile(
    r"^\s*((第[一二三四五六七八九十百千万\d]+[编章节条款项])|([一二三四五六七八九十]+、)|(\d+(\.\d+){0,4}\s+)|([（(][一二三四五六七八九十\d]+[)）])|#{1,6}\s+)"
)
PAGE_RE = re.compile(r"^\[PAGE\s+(\d+)]")
KEYWORD_RE = re.compile(
    r"(第[一二三四五六七八九十百千万\d]+[编章节条款项]|20\d{2}|19\d{2}|\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s*(?:亿元|万元|元|倍|日|天|个月|年|个工作日)|《[^》]{2,60}》|[A-Za-z][A-Za-z0-9_\-]{1,30}|[\u4e00-\u9fff]{2,8})"
)
ARTICLE_RE = re.compile(r"^\s*第[一二三四五六七八九十百千万\d]+条")
INSURANCE_BOUNDARY_RE = re.compile(
    r"^\s*(保险责任|责任免除|除外责任|等待期|犹豫期|宽限期|身故保险金|退保|现金价值|保单贷款|免赔额|赔偿处理|给付|领取|释义|如何申请)"
)
TABLE_LINE_RE = re.compile(r"^\s*(\|.*\||[-+]{3,}|表\s*\d+|单位[:：])")


def extract_keywords(text: str, max_keywords: int = 60) -> list[str]:
    seen: set[str] = set()
    keywords: list[str] = []
    for match in KEYWORD_RE.findall(text):
        token = match[0] if isinstance(match, tuple) else match
        token = token.strip()
        if token and token not in seen:
            seen.add(token)
            keywords.append(token)
            if len(keywords) >= max_keywords:
                break
    return keywords


class Chunker:
    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None):
        self.max_chars = int(config.get("chunk_max_chars", 1800))
        self.overlap_chars = int(config.get("chunk_overlap_chars", 260))
        self.min_chunk_chars = int(config.get("min_chunk_chars", 80))
        self.domain_config = config.get("domain_chunking", {})
        self.logger = logger or logging.getLogger(__name__)

    def chunk_document(self, document: Document) -> list[Chunk]:
        max_chars, overlap_chars = self._limits_for_domain(document.domain)
        blocks = self._paragraph_blocks(document.text)
        chunks: list[Chunk] = []
        buffer: list[dict[str, Any]] = []
        buffer_len = 0
        section_path: list[str] = []

        for original_block in blocks:
            split_blocks = self._split_large_block(original_block, max_chars, document.domain)
            for block in split_blocks:
                text = block["text"]
                is_boundary = self._is_domain_boundary(document.domain, text)
                if (SECTION_RE.match(text) or is_boundary) and len(text) <= 160:
                    section_path = self._update_section_path(section_path, text)
                block["section_path"] = list(section_path)

                if buffer and is_boundary and buffer_len >= self.min_chunk_chars:
                    chunks.append(self._make_chunk(document, buffer, len(chunks)))
                    buffer = self._overlap_tail(buffer, overlap_chars)
                    buffer_len = sum(len(x["text"]) + 2 for x in buffer)

                if buffer and buffer_len + len(text) > max_chars:
                    chunks.append(self._make_chunk(document, buffer, len(chunks)))
                    buffer = self._overlap_tail(buffer, overlap_chars)
                    buffer_len = sum(len(x["text"]) + 2 for x in buffer)

                buffer.append(block)
                buffer_len += len(text) + 2

        if buffer and buffer_len >= self.min_chunk_chars:
            chunks.append(self._make_chunk(document, buffer, len(chunks)))
        return chunks

    def _limits_for_domain(self, domain: str) -> tuple[int, int]:
        conf = self.domain_config.get(domain, {})
        return int(conf.get("max_chars", self.max_chars)), int(conf.get("overlap_chars", self.overlap_chars))

    def _paragraph_blocks(self, text: str) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        page = 1
        cursor = 0
        for raw in re.split(r"\n\s*\n", text):
            stripped = raw.strip()
            if not stripped:
                cursor += len(raw) + 2
                continue
            page_match = PAGE_RE.match(stripped)
            if page_match:
                page = int(page_match.group(1))
                stripped = PAGE_RE.sub("", stripped).strip()
                if not stripped:
                    cursor += len(raw) + 2
                    continue
            block_type = "table" if TABLE_LINE_RE.match(stripped) else "text"
            blocks.append({"text": stripped, "page": page, "char_start": cursor, "char_end": cursor + len(raw), "type": block_type})
            cursor += len(raw) + 2
        return blocks

    def _split_large_block(self, block: dict[str, Any], max_chars: int, domain: str) -> list[dict[str, Any]]:
        text = block["text"]
        if len(text) <= int(max_chars * 1.15):
            return [block]

        prefer_lines = block.get("type") == "table" or domain in {"financial_reports", "financial_contracts"}
        units = [line for line in text.splitlines() if line.strip()] if prefer_lines else re.split(r"(?<=[。！？；;])", text)
        if len(units) <= 1:
            units = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

        split_blocks: list[dict[str, Any]] = []
        current: list[str] = []
        current_len = 0
        offset = int(block["char_start"])
        cursor = offset
        hard_limit = max(self.min_chunk_chars, max_chars)

        for unit in units:
            unit = unit.strip()
            if not unit:
                continue
            if current and current_len + len(unit) + 1 > hard_limit:
                split_text = "\n".join(current) if prefer_lines else "".join(current)
                split_blocks.append({**block, "text": split_text, "char_start": cursor, "char_end": cursor + len(split_text)})
                cursor += len(split_text)
                current = []
                current_len = 0
            current.append(unit)
            current_len += len(unit) + 1

        if current:
            split_text = "\n".join(current) if prefer_lines else "".join(current)
            split_blocks.append({**block, "text": split_text, "char_start": cursor, "char_end": cursor + len(split_text)})

        return split_blocks or [block]

    def _update_section_path(self, path: list[str], heading: str) -> list[str]:
        heading = heading.strip().lstrip("#").strip()
        if heading.startswith("第") and ("编" in heading or "章" in heading or "节" in heading):
            return [heading]
        if heading.startswith("第") and ("条" in heading or "款" in heading or "项" in heading):
            return (path[:1] if path else []) + [heading]
        if INSURANCE_BOUNDARY_RE.match(heading):
            return (path[:1] if path else []) + [heading]
        if re.match(r"^\d+\.\d+", heading):
            return (path[:1] if path else []) + [heading]
        return (path[:1] if path else []) + [heading]

    def _is_domain_boundary(self, domain: str, text: str) -> bool:
        text = text.strip()
        if domain == "regulatory":
            return bool(ARTICLE_RE.match(text))
        if domain == "insurance":
            return bool(ARTICLE_RE.match(text) or INSURANCE_BOUNDARY_RE.match(text))
        if domain in {"financial_contracts", "financial_reports", "research"}:
            return bool(SECTION_RE.match(text) and len(text) <= 120)
        return bool(SECTION_RE.match(text) and len(text) <= 120)

    def _overlap_tail(self, blocks: list[dict[str, Any]], overlap_chars: int) -> list[dict[str, Any]]:
        tail: list[dict[str, Any]] = []
        total = 0
        for block in reversed(blocks):
            tail.insert(0, block)
            total += len(block["text"]) + 2
            if total >= overlap_chars:
                break
        return tail

    def _make_chunk(self, document: Document, blocks: list[dict[str, Any]], index: int) -> Chunk:
        text = "\n\n".join(block["text"] for block in blocks).strip()
        page_start = min(int(block["page"]) for block in blocks)
        page_end = max(int(block["page"]) for block in blocks)
        char_start = min(int(block["char_start"]) for block in blocks)
        char_end = max(int(block["char_end"]) for block in blocks)
        section_path = blocks[-1].get("section_path") or []
        chunk_id = f"{document.doc_id}__{index:04d}_{stable_id(document.doc_id, index, char_start)}"
        return Chunk(
            chunk_id=chunk_id,
            doc_id=document.doc_id,
            domain=document.domain,
            title=document.title,
            page_start=page_start,
            page_end=page_end,
            section_path=section_path,
            text=text,
            keywords=extract_keywords(text),
            char_start=char_start,
            char_end=char_end,
        )
