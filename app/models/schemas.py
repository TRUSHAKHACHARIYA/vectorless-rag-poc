from enum import Enum

from pydantic import BaseModel, Field


class FileStatus(str, Enum):
    INDEXED = "indexed"
    PENDING = "pending"
    FAILED = "failed"


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: str
    text: str
    bm25_score: float
    chunk_index: int
    page_num: int | None = None


class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    num_chunks: int
    status: FileStatus


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    status: FileStatus
    message: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentMeta]
    total: int


class DeleteResponse(BaseModel):
    doc_id: str
    message: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = None
    doc_ids: list[str] | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    docs_searched: int
    chunks_searched: int
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    model_used: str


class ChunkMeta(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    page_num: int | None = None
