"""
Document parser — supports PDF, TXT, MD, DOCX.
Returns raw text + per-page metadata.
"""
from pathlib import Path
from dataclasses import dataclass

from app.core.exceptions import DocumentParseError, UnsupportedFileTypeError
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedPage:
    page_num: int
    text: str


@dataclass
class ParsedDocument:
    filename: str
    file_type: str
    pages: list[ParsedPage]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def total_chars(self) -> int:
        return len(self.full_text)


class DocumentParser:
    SUPPORTED = {".pdf", ".txt", ".md", ".docx"}

    def parse(self, file_path: Path) -> ParsedDocument:
        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {ext}",
                {"supported": list(self.SUPPORTED)},
            )

        logger.info(f"Parsing {file_path.name} ({ext})")

        try:
            if ext == ".pdf":
                return self._parse_pdf(file_path)
            elif ext in {".txt", ".md"}:
                return self._parse_text(file_path)
            elif ext == ".docx":
                return self._parse_docx(file_path)
        except (UnsupportedFileTypeError, DocumentParseError):
            raise
        except Exception as e:
            raise DocumentParseError(f"Failed to parse {file_path.name}: {e}")

    # ── Parsers ────────────────────────────────────────────────────────────────

    def _parse_pdf(self, path: Path) -> ParsedDocument:
        import fitz  # pymupdf

        doc = fitz.open(str(path))
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append(ParsedPage(page_num=i + 1, text=text))
        doc.close()

        if not pages:
            raise DocumentParseError(f"PDF {path.name} contains no extractable text (scanned?)")

        return ParsedDocument(filename=path.name, file_type="pdf", pages=pages)

    def _parse_text(self, path: Path) -> ParsedDocument:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            raise DocumentParseError(f"{path.name} is empty")
        return ParsedDocument(
            filename=path.name,
            file_type=path.suffix.lstrip("."),
            pages=[ParsedPage(page_num=1, text=text)],
        )

    def _parse_docx(self, path: Path) -> ParsedDocument:
        from docx import Document

        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if not text:
            raise DocumentParseError(f"DOCX {path.name} contains no text")
        return ParsedDocument(
            filename=path.name,
            file_type="docx",
            pages=[ParsedPage(page_num=1, text=text)],
        )
