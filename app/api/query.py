"""
/query — BM25 retrieval + Ollama generation.
Supports both standard JSON and streaming (SSE) responses.
"""
import time
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.core.dependencies import get_bm25_index, get_ollama_service
from app.core.exceptions import IndexNotFoundError, OllamaConnectionError
from app.core.logger import get_logger
from app.models.schemas import QueryRequest, QueryResponse
from app.services.bm25_index import BM25IndexManager
from app.services.ollama_service import OllamaService

router = APIRouter(prefix="/query", tags=["Query"])
logger = get_logger(__name__)


@router.post("", response_model=QueryResponse)
def query(
    request: QueryRequest,
    index: BM25IndexManager = Depends(get_bm25_index),
    llm: OllamaService = Depends(get_ollama_service),
):
    """
    Ask a question over indexed documents.
    1. BM25 retrieves the most relevant chunks (no embeddings).
    2. Ollama (local LLM) generates an answer from those chunks.
    """
    t_total = time.perf_counter()

    # ── Retrieve ──────────────────────────────────────────────────────────────
    try:
        chunks, retrieval_ms = index.retrieve(
            query=request.query,
            top_k=request.top_k,
            doc_ids=request.doc_ids,
        )
    except IndexNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    logger.info(
        f"Query: '{request.query[:80]}' → {len(chunks)} chunks retrieved in {retrieval_ms:.1f}ms"
    )

    # ── Generate ──────────────────────────────────────────────────────────────
    try:
        answer, gen_ms = llm.generate(request.query, chunks)
    except OllamaConnectionError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    total_ms = (time.perf_counter() - t_total) * 1000

    return QueryResponse(
        query=request.query,
        answer=answer,
        retrieved_chunks=chunks,
        docs_searched=index.total_documents,
        chunks_searched=index.total_chunks,
        retrieval_time_ms=round(retrieval_ms, 2),
        generation_time_ms=round(gen_ms, 2),
        total_time_ms=round(total_ms, 2),
        model_used=llm.model,
    )


@router.post("/stream")
async def query_stream(
    request: QueryRequest,
    index: BM25IndexManager = Depends(get_bm25_index),
    llm: OllamaService = Depends(get_ollama_service),
):
    """
    Streaming version — returns SSE tokens as they arrive from Ollama.
    Useful for UI integration.
    """
    try:
        chunks, _ = index.retrieve(
            query=request.query,
            top_k=request.top_k,
            doc_ids=request.doc_ids,
        )
    except IndexNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    async def event_stream():
        # First send retrieved sources as metadata
        import json
        sources = [
            {"filename": c.filename, "page": c.page_num, "score": c.bm25_score}
            for c in chunks
        ]
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"

        # Then stream tokens
        async for token in llm.stream_generate(request.query, chunks):
            yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/retrieve-only", response_model=list)
def retrieve_only(
    request: QueryRequest,
    index: BM25IndexManager = Depends(get_bm25_index),
):
    """
    BM25 retrieval only — no LLM generation.
    Useful for debugging retrieval quality.
    """
    try:
        chunks, retrieval_ms = index.retrieve(
            query=request.query,
            top_k=request.top_k,
            doc_ids=request.doc_ids,
        )
    except IndexNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    return [
        {**c.model_dump(), "retrieval_time_ms": round(retrieval_ms, 2)}
        for c in chunks
    ]
