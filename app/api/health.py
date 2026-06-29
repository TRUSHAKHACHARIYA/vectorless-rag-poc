from fastapi import APIRouter, Depends

from app.core.config import get_settings
from app.core.dependencies import get_bm25_index, get_ollama_service
from app.services.bm25_index import BM25IndexManager
from app.services.ollama_service import OllamaService

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health(
    index: BM25IndexManager = Depends(get_bm25_index),
    llm: OllamaService = Depends(get_ollama_service),
):
    cfg = get_settings()
    ollama_ok = await llm.is_reachable()

    return {
        "status": "healthy" if ollama_ok else "degraded",
        "ollama": {
            "reachable": ollama_ok,
            "model": cfg.ollama_model,
            "base_url": cfg.ollama_base_url,
        },
        "index": {
            "documents": index.total_documents,
            "chunks": index.total_chunks,
        },
    }
