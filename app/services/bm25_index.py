"""
BM25IndexManager — builds and queries a BM25 index over document chunks.
No embeddings. No vector DB. Pure keyword retrieval with Okapi BM25.

Architecture:
  - One global index across all docs (fast, simple for POC)
  - Per-doc re-indexing on add/delete
  - In-memory store (upgrade path: persist to Redis or disk)
"""
import re
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock

from rank_bm25 import BM25Okapi

from app.models.schemas import RetrievedChunk, DocumentMeta, FileStatus
from app.services.chunker import Chunk
from app.core.config import get_settings
from app.core.exceptions import IndexNotFoundError
from app.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IndexedDocument:
    doc_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    chunks: list[Chunk]
    meta: DocumentMeta


class BM25IndexManager:
    """Thread-safe BM25 index over all ingested document chunks."""

    def __init__(self):
        self._lock = Lock()
        self._documents: dict[str, IndexedDocument] = {}   # doc_id → doc
        self._bm25: BM25Okapi | None = None
        self._flat_chunks: list[Chunk] = []                # flat list for retrieval
        self._tokenized_corpus: list[list[str]] = []

    # ── Tokenization ──────────────────────────────────────────────────────────

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Lowercase, remove punctuation, split on whitespace."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        # Remove single-char tokens for better BM25 precision
        return [t for t in tokens if len(t) > 1]

    # ── Index Management ──────────────────────────────────────────────────────

    def add_document(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        file_size_bytes: int,
        chunks: list[Chunk],
    ) -> DocumentMeta:
        """Add a document's chunks to the index. Rebuilds BM25 after insertion."""
        if not chunks:
            raise ValueError(f"No chunks provided for {filename}")

        meta = DocumentMeta(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            num_chunks=len(chunks),
            status=FileStatus.INDEXED,
        )
        indexed_doc = IndexedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            chunks=chunks,
            meta=meta,
        )

        with self._lock:
            self._documents[doc_id] = indexed_doc
            self._rebuild_index()

        logger.info(f"Indexed '{filename}' ({len(chunks)} chunks). Total docs: {len(self._documents)}")
        return meta

    def delete_document(self, doc_id: str) -> None:
        """Remove a document and rebuild the index."""
        with self._lock:
            if doc_id not in self._documents:
                raise IndexNotFoundError(f"Document {doc_id} not found in index")
            del self._documents[doc_id]
            self._rebuild_index()
        logger.info(f"Deleted doc {doc_id}. Remaining docs: {len(self._documents)}")

    def _rebuild_index(self) -> None:
        """Rebuild BM25 from scratch over all current chunks. Called under lock."""
        self._flat_chunks = []
        for doc in self._documents.values():
            self._flat_chunks.extend(doc.chunks)

        if not self._flat_chunks:
            self._bm25 = None
            self._tokenized_corpus = []
            return

        self._tokenized_corpus = [self.tokenize(c.text) for c in self._flat_chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.debug(f"BM25 index rebuilt: {len(self._flat_chunks)} total chunks")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> tuple[list[RetrievedChunk], float]:
        """
        Retrieve top-k chunks for a query using BM25.

        Args:
            query:   Natural language question.
            top_k:   Number of chunks to return.
            doc_ids: Optional whitelist of doc IDs to restrict search.

        Returns:
            (list of RetrievedChunk, retrieval_time_ms)
        """
        cfg = get_settings()
        top_k = top_k or cfg.bm25_top_k

        if self._bm25 is None or not self._flat_chunks:
            raise IndexNotFoundError("No documents indexed yet. Upload documents first.")

        t0 = time.perf_counter()
        tokenized_query = self.tokenize(query)

        # BM25 scores for all chunks
        scores = self._bm25.get_scores(tokenized_query)

        # Build (score, chunk) pairs, filtering by doc_ids if given
        scored: list[tuple[float, Chunk]] = []
        for score, chunk in zip(scores, self._flat_chunks):
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            scored.append((float(score), chunk))

        # Sort descending, take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        # Drop zero-score results (no keyword overlap at all)
        top = [(s, c) for s, c in top if s > 0.0]

        retrieval_ms = (time.perf_counter() - t0) * 1000

        results = [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                filename=self._documents[chunk.doc_id].filename,
                text=chunk.text,
                bm25_score=round(score, 4),
                chunk_index=chunk.chunk_index,
                page_num=chunk.page_num,
            )
            for score, chunk in top
        ]

        logger.debug(
            f"BM25 retrieved {len(results)} chunks for query '{query[:60]}' in {retrieval_ms:.1f}ms"
        )
        return results, retrieval_ms

    # ── Metadata ──────────────────────────────────────────────────────────────

    def list_documents(self) -> list[DocumentMeta]:
        return [d.meta for d in self._documents.values()]

    def get_document(self, doc_id: str) -> DocumentMeta:
        if doc_id not in self._documents:
            raise IndexNotFoundError(f"Document {doc_id} not found")
        return self._documents[doc_id].meta

    @property
    def total_chunks(self) -> int:
        return len(self._flat_chunks)

    @property
    def total_documents(self) -> int:
        return len(self._documents)
