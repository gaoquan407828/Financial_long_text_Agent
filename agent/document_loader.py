from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from .schema import Document, Question
from .utils import compact_whitespace, command_exists, ensure_dir, load_json, resolve_path, run_command


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".json", ".html", ".htm"}


def infer_domain(path: Path, raw_root: Path) -> str:
    try:
        rel = path.relative_to(raw_root)
        return rel.parts[0]
    except ValueError:
        return path.parent.name


def infer_doc_id(path: Path) -> str:
    return path.stem


def infer_title(path: Path) -> str:
    title = path.stem
    title = re.sub(r"[_\-]+", " ", title)
    return title.strip()


def discover_raw_documents(raw_root: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in sorted(raw_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append(
                {
                    "doc_id": infer_doc_id(path),
                    "domain": infer_domain(path, raw_root),
                    "title": infer_title(path),
                    "path": str(path),
                    "source_type": path.suffix.lower().lstrip("."),
                }
            )
    return docs


def load_questions(path: Path) -> list[Question]:
    files: list[Path]
    if path.is_dir():
        files = sorted(path.glob("*.json"))
    else:
        files = [path]
    questions: list[Question] = []
    for file_path in files:
        data = load_json(file_path)
        if isinstance(data, dict):
            data = data.get("questions", [])
        questions.extend(Question.from_dict(item) for item in data)
    return questions


class DocumentLoader:
    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.pdf_config = config.get("pdf", {})

    def load(self, metadata: dict[str, Any]) -> Document:
        path = Path(metadata["path"])
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text, pages, parse_meta = self._load_pdf(path, metadata)
        elif suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            pages = [{"page": 1, "text": text, "source": "text"}]
            parse_meta = {"parser": "plain_text"}
        elif suffix == ".json":
            text, pages, parse_meta = self._load_json_document(path)
        elif suffix in {".html", ".htm"}:
            text = self._load_html(path)
            pages = [{"page": 1, "text": text, "source": "html"}]
            parse_meta = {"parser": "beautifulsoup"}
        else:
            raise ValueError(f"Unsupported document type: {path}")

        text = compact_whitespace(text)
        return Document(
            doc_id=str(metadata["doc_id"]),
            domain=str(metadata["domain"]),
            title=str(metadata.get("title") or infer_title(path)),
            path=str(path),
            source_type=suffix.lstrip("."),
            text=text,
            pages=pages,
            metadata={**metadata, **parse_meta},
        )

    def _load_json_document(self, path: Path) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, dict):
            if "text" in data:
                text = str(data["text"])
            else:
                text = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            text = json.dumps(data, ensure_ascii=False, indent=2)
        return text, [{"page": 1, "text": text, "source": "json"}], {"parser": "json"}

    def _load_html(self, path: Path) -> str:
        raw = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return soup.get_text("\n")

    def _load_pdf(self, path: Path, metadata: dict[str, Any]) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        parser_order = self.pdf_config.get("parser_order") or ["pymupdf4llm", "pdftotext", "pypdf"]
        pages: list[dict[str, Any]] = []
        parser = "unparsed"
        for parser_name in parser_order:
            parser_name = str(parser_name).lower()
            if parser_name == "pymupdf4llm":
                pages = self._pdf_markdown_with_pymupdf4llm(path, metadata)
                parser = "pymupdf4llm"
            elif parser_name == "pdftotext" and command_exists("pdftotext"):
                pages = self._pdf_text_with_pdftotext(path)
                parser = "pdftotext"
            elif parser_name in {"pypdf", "python_pdf"}:
                pages = self._pdf_text_with_python(path)
                parser = "pypdf"
            else:
                pages = []

            if self._has_enough_pdf_text(pages):
                break
            if pages:
                self.logger.info("%s produced little text for %s; trying next parser", parser, path)

        min_chars = int(self.pdf_config.get("min_text_chars_per_page", 40))
        needs_ocr = [p for p in pages if len(p.get("text", "").strip()) < min_chars]
        if needs_ocr and self.pdf_config.get("ocr_enabled", True):
            ocr_pages = self._ocr_pdf_pages(path, pages, min_chars=min_chars)
            if ocr_pages:
                pages = ocr_pages
                parser = f"{parser}+ocr"

        full_text = "\n\n".join(f"[PAGE {p['page']}]\n{p['text']}" for p in pages if p.get("text", "").strip())
        if self.pdf_config.get("save_markdown", True):
            self._save_pdf_markdown(str(metadata["doc_id"]), pages)
        return full_text, pages, {"parser": parser, "page_count": len(pages)}

    def _has_enough_pdf_text(self, pages: list[dict[str, Any]]) -> bool:
        min_chars = int(self.pdf_config.get("min_text_chars_per_page", 40))
        if not pages:
            return False
        total_text = sum(len(p.get("text", "").strip()) for p in pages)
        non_empty = sum(1 for p in pages if len(p.get("text", "").strip()) >= min_chars)
        return total_text >= min_chars and non_empty >= max(1, min(2, len(pages)))

    def _pdf_markdown_with_pymupdf4llm(self, path: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            import pymupdf4llm
        except Exception as exc:
            self.logger.info("pymupdf4llm is unavailable for %s: %s", path, exc)
            return []

        kwargs: dict[str, Any] = {}
        if bool(self.pdf_config.get("pymupdf4llm_page_chunks", True)):
            kwargs["page_chunks"] = True
        if bool(self.pdf_config.get("pymupdf4llm_use_ocr", False)):
            kwargs["use_ocr"] = True
        if bool(self.pdf_config.get("pymupdf4llm_force_ocr", False)):
            kwargs["force_ocr"] = True
        try:
            markdown = pymupdf4llm.to_markdown(str(path), **kwargs)
        except TypeError as exc:
            self.logger.warning("pymupdf4llm kwargs unsupported for %s (%s); retrying minimal call", path, exc)
            try:
                markdown = pymupdf4llm.to_markdown(str(path))
            except Exception as inner_exc:
                self.logger.warning("pymupdf4llm failed for %s: %s", path, inner_exc)
                return []
        except Exception as exc:
            self.logger.warning("pymupdf4llm failed for %s: %s", path, exc)
            return []

        pages = self._normalize_pymupdf4llm_output(markdown)
        return pages

    def _normalize_pymupdf4llm_output(self, markdown: Any) -> list[dict[str, Any]]:
        if isinstance(markdown, str):
            return [{"page": 1, "text": compact_whitespace(markdown), "source": "pymupdf4llm_markdown"}]
        if not isinstance(markdown, list):
            return [{"page": 1, "text": compact_whitespace(str(markdown)), "source": "pymupdf4llm_markdown"}]

        pages: list[dict[str, Any]] = []
        for idx, item in enumerate(markdown, start=1):
            if isinstance(item, dict):
                metadata = item.get("metadata") or {}
                page_no = (
                    item.get("page")
                    or item.get("page_number")
                    or metadata.get("page")
                    or metadata.get("page_number")
                    or idx
                )
                text = item.get("text") or item.get("markdown") or item.get("content") or ""
                pages.append(
                    {
                        "page": int(page_no),
                        "text": compact_whitespace(str(text)),
                        "source": "pymupdf4llm_markdown",
                        "metadata": metadata,
                    }
                )
            else:
                pages.append(
                    {
                        "page": idx,
                        "text": compact_whitespace(str(item)),
                        "source": "pymupdf4llm_markdown",
                    }
                )
        return pages

    def _save_pdf_markdown(self, doc_id: str, pages: list[dict[str, Any]]) -> None:
        markdown_dir = ensure_dir(resolve_path(self.config.get("markdown_dir", "processed_data/markdown")))
        md_path = markdown_dir / f"{doc_id}.md"
        blocks = []
        for page in pages:
            text = page.get("text", "").strip()
            if text:
                blocks.append(f"<!-- page: {page.get('page', '')} -->\n\n{text}")
        md_path.write_text("\n\n".join(blocks), encoding="utf-8")

    def _pdf_text_with_pdftotext(self, path: Path) -> list[dict[str, Any]]:
        result = run_command(["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"], timeout=180)
        if result.returncode != 0:
            self.logger.warning("pdftotext failed for %s: %s", path, result.stderr.strip())
            return self._pdf_text_with_python(path)
        raw_pages = result.stdout.split("\f")
        if raw_pages and not raw_pages[-1].strip():
            raw_pages = raw_pages[:-1]
        page_count = self._pdf_page_count(path)
        if page_count and len(raw_pages) < page_count:
            raw_pages.extend([""] * (page_count - len(raw_pages)))
        return [
            {"page": idx + 1, "text": compact_whitespace(text), "source": "pdftotext"}
            for idx, text in enumerate(raw_pages)
        ]

    def _pdf_page_count(self, path: Path) -> int:
        if not command_exists("pdfinfo"):
            return 0
        result = run_command(["pdfinfo", str(path)], timeout=60)
        if result.returncode != 0:
            return 0
        match = re.search(r"Pages:\s+(\d+)", result.stdout)
        return int(match.group(1)) if match else 0

    def _pdf_text_with_python(self, path: Path) -> list[dict[str, Any]]:
        try:
            from pypdf import PdfReader
        except Exception as exc:
            self.logger.warning("pypdf is unavailable for %s: %s", path, exc)
            return [{"page": 1, "text": "", "source": "missing_pdf_parser"}]

        reader = PdfReader(str(path))
        pages: list[dict[str, Any]] = []
        for idx, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                self.logger.warning("pypdf page extraction failed for %s page %s: %s", path, idx + 1, exc)
                text = ""
            pages.append({"page": idx + 1, "text": compact_whitespace(text), "source": "pypdf"})
        return pages

    def _ocr_pdf_pages(
        self,
        path: Path,
        current_pages: list[dict[str, Any]],
        min_chars: int,
    ) -> list[dict[str, Any]]:
        if not command_exists("pdftoppm"):
            self.logger.warning("OCR skipped for %s because pdftoppm is not available", path)
            return []
        tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")
        if not command_exists(tesseract_cmd):
            self.logger.warning("OCR skipped for %s because tesseract is not available", path)
            return []

        max_ocr_pages = int(self.pdf_config.get("max_ocr_pages", 0) or 0)
        lang = str(self.pdf_config.get("ocr_lang", os.getenv("OCR_LANG", "chi_sim+eng")))
        dpi = int(self.pdf_config.get("ocr_dpi", 220))
        if not current_pages:
            page_count = self._pdf_page_count(path)
            current_pages = [{"page": page, "text": "", "source": "empty"} for page in range(1, page_count + 1)]
        pages_to_ocr = [
            p["page"] for p in current_pages if len(p.get("text", "").strip()) < min_chars
        ]
        if max_ocr_pages > 0:
            pages_to_ocr = pages_to_ocr[:max_ocr_pages]
        if not pages_to_ocr:
            return []

        page_map = {p["page"]: dict(p) for p in current_pages}
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = Path(tmpdir) / "page"
            result = run_command(["pdftoppm", "-r", str(dpi), "-png", str(path), str(prefix)], timeout=300)
            if result.returncode != 0:
                self.logger.warning("pdftoppm failed for %s: %s", path, result.stderr.strip())
                return []
            for page_no in pages_to_ocr:
                image_path = Path(tmpdir) / f"page-{page_no}.png"
                if not image_path.exists():
                    image_path = Path(tmpdir) / f"page-{page_no:06d}.png"
                if not image_path.exists():
                    continue
                ocr = run_command([tesseract_cmd, str(image_path), "stdout", "-l", lang], timeout=180)
                if ocr.returncode == 0 and ocr.stdout.strip():
                    page_map[page_no] = {
                        "page": page_no,
                        "text": compact_whitespace(ocr.stdout),
                        "source": "tesseract_ocr",
                    }
                else:
                    self.logger.warning("tesseract failed for %s page %s: %s", path, page_no, ocr.stderr.strip())
        return [page_map[k] for k in sorted(page_map)]
