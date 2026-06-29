"""
Simple word-boundary chunker with overlap.
No LangChain dependency here — pure Python for speed & testability.
"""
import re
import uuid
from dataclasses import dataclass

from app.services.document_parser import ParsedDocument
from app.models.schemas import ChunkMeta
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    page_num: int | None = None


class TextChunker:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        cfg = get_settings()
        self.chunk_size = chunk_size or cfg.bm25_chunk_size
        self.chunk_overlap = chunk_overlap or cfg.bm25_chunk_overlap

    def chunk_document(self, parsed_doc: ParsedDocument, doc_id: str) -> list[Chunk]:
        """Chunk a parsed document respecting page boundaries."""
        all_chunks: list[Chunk] = []
        chunk_index = 0

        for page in parsed_doc.pages:
            page_chunks = self._chunk_text(page.text, doc_id, chunk_index, page.page_num)
            all_chunks.extend(page_chunks)
            chunk_index += len(page_chunks)

        logger.info(
            f"Chunked {parsed_doc.filename} → {len(all_chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks

    def _chunk_text(
        self,
        text: str,
        doc_id: str,
        start_index: int,
        page_num: int | None,
    ) -> list[Chunk]:
        """Split text into overlapping chunks at word boundaries."""
        # Normalize whitespace but preserve paragraph breaks
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        chunk_index = start_index

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Extend to word boundary
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        char_start=start,
                        char_end=end,
                        page_num=page_num,
                    )
                )
                chunk_index += 1

            # Move forward with overlap
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return chunks

    @staticmethod
    def to_schema(chunk: Chunk) -> ChunkMeta:
        return ChunkMeta(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=chunk.text,
            chunk_index=chunk.chunk_index,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            page_num=chunk.page_num,
        )
