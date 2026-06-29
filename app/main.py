"""
Vectorless RAG — FastAPI Application
Zero embeddings. Zero vector DB. BM25 + Ollama.
"""
import utils.session
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api import ingest, query, health
from app.core.config import get_settings
from app.core.logger import setup_logger, get_logger
from app.core.exceptions import VectorlessRAGException

setup_logger()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    logger.info(f"🚀 Vectorless RAG starting | model={cfg.ollama_model} | env={cfg.app_env}")
    yield
    logger.info("Vectorless RAG shutting down")


def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="Vectorless RAG",
        description=(
            "FAQ / document Q&A using BM25 keyword retrieval + Ollama local LLM. "
            "No embeddings. No vector DB. Zero cost."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if cfg.app_env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(VectorlessRAGException)
    async def rag_exception_handler(request: Request, exc: VectorlessRAGException):
        logger.error(f"RAG Error: {exc.message} | {exc.details}")
        return JSONResponse(
            status_code=500,
            content={"error": exc.message, "details": exc.details},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "details": str(exc)},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(ingest.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")

    static_dir = Path(__file__).parent.parent / "static"

    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()
