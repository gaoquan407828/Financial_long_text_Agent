from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .chunker import Chunker
from .document_loader import DocumentLoader, discover_raw_documents
from .schema import Document
from .utils import ensure_dir, resolve_path, write_json, write_jsonl


class Preprocessor:
    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.raw_root = resolve_path(config["raw_data_dir"])
        self.documents_dir = ensure_dir(resolve_path(config["documents_dir"]))
        self.chunks_dir = ensure_dir(resolve_path(config["chunks_dir"]))
        self.metadata_dir = ensure_dir(resolve_path(config["metadata_dir"]))
        self.loader = DocumentLoader(config, self.logger)
        self.chunker = Chunker(config, self.logger)

    def run(
        self,
        domains: set[str] | None = None,
        limit: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        metadata = discover_raw_documents(self.raw_root)
        if domains:
            metadata = [m for m in metadata if m["domain"] in domains]
        if limit:
            metadata = metadata[:limit]
        processed_docs: list[dict[str, Any]] = []
        total_chunks = 0
        for idx, item in enumerate(metadata, start=1):
            doc_id = item["doc_id"]
            doc_path = self.documents_dir / f"{doc_id}.json"
            chunk_path = self.chunks_dir / f"{doc_id}.jsonl"
            if doc_path.exists() and chunk_path.exists() and not force:
                self.logger.info("skip existing document %s (%s/%s)", doc_id, idx, len(metadata))
                chunk_count = self._count_jsonl_lines(chunk_path)
                total_chunks += chunk_count
                processed_docs.append(
                    {
                        **item,
                        "processed_document": str(doc_path),
                        "chunks": str(chunk_path),
                        "chunk_count": chunk_count,
                    }
                )
                continue

            self.logger.info("processing %s (%s/%s)", doc_id, idx, len(metadata))
            try:
                document = self.loader.load(item)
                chunks = self.chunker.chunk_document(document)
                write_json(doc_path, document.to_dict())
                write_jsonl(chunk_path, [chunk.to_dict() for chunk in chunks])
                total_chunks += len(chunks)
                processed_docs.append(
                    {
                        **item,
                        "processed_document": str(doc_path),
                        "chunks": str(chunk_path),
                        "chunk_count": len(chunks),
                        "char_count": len(document.text),
                    }
                )
            except Exception as exc:
                self.logger.exception("failed to process %s: %s", item["path"], exc)
                processed_docs.append({**item, "error": str(exc)})

        manifest = {
            "raw_root": str(self.raw_root),
            "document_count": len(processed_docs),
            "chunk_count": total_chunks,
            "documents": processed_docs,
        }
        write_json(self.metadata_dir / "documents.json", manifest)
        return manifest

    def _count_jsonl_lines(self, path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def load_processed_documents(self) -> list[Document]:
        docs: list[Document] = []
        for path in sorted(self.documents_dir.glob("*.json")):
            from .utils import load_json

            docs.append(Document.from_dict(load_json(path)))
        return docs
