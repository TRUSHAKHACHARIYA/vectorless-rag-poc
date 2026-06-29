from functools import lru_cache

from app.services.bm25_index import BM25IndexManager
from app.services.chunker import TextChunker
from app.services.document_parser import DocumentParser
from app.services.ollama_service import OllamaService


@lru_cache
def get_bm25_index() -> BM25IndexManager:
    return BM25IndexManager()


@lru_cache
def get_document_parser() -> DocumentParser:
    return DocumentParser()


@lru_cache
def get_chunker() -> TextChunker:
    return TextChunker()


@lru_cache
def get_ollama_service() -> OllamaService:
    return OllamaService()
