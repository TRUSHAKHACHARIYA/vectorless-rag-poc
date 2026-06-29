"""
OllamaService — wraps the Ollama API for generation.
Handles connection checks, prompt building, streaming, and error recovery.
"""
import time
from typing import AsyncIterator

import httpx
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from app.models.schemas import RetrievedChunk
from app.core.config import get_settings
from app.core.exceptions import OllamaConnectionError
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Prompt Template ───────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer, say "I don't have enough information to answer this question."
Do not make up or infer information beyond what is explicitly stated in the context.

Context:
{context}

Question: {question}

Answer:""",
)


class OllamaService:
    def __init__(self):
        cfg = get_settings()
        self.model = cfg.ollama_model
        self.base_url = cfg.ollama_base_url

        self._llm = OllamaLLM(
            model=self.model,
            base_url=self.base_url,
            temperature=cfg.ollama_temperature,
            num_ctx=cfg.ollama_num_ctx,
        )
        self._chain = RAG_PROMPT | self._llm

    # ── Health check ──────────────────────────────────────────────────────────

    async def is_reachable(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False

    # ── Context builder ───────────────────────────────────────────────────────

    @staticmethod
    def build_context(chunks: list[RetrievedChunk], max_chars: int = 3000) -> str:
        """
        Build a numbered context string from retrieved chunks.
        Respects max_chars to avoid overflowing context window.
        """
        parts: list[str] = []
        total = 0
        for i, chunk in enumerate(chunks, 1):
            header = f"[Source {i}: {chunk.filename}"
            if chunk.page_num:
                header += f", page {chunk.page_num}"
            header += "]"
            block = f"{header}\n{chunk.text}"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n---\n\n".join(parts)

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> tuple[str, float]:
        """Synchronous generation. Returns (answer, generation_time_ms)."""
        if not chunks:
            return "No relevant context found to answer your question.", 0.0

        context = self.build_context(chunks)
        t0 = time.perf_counter()

        try:
            answer = self._chain.invoke({"context": context, "question": query})
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise OllamaConnectionError(f"Ollama generation failed: {e}")

        gen_ms = (time.perf_counter() - t0) * 1000
        logger.debug(f"Generated answer in {gen_ms:.0f}ms ({len(answer)} chars)")
        return answer.strip(), gen_ms

    async def stream_generate(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> AsyncIterator[str]:
        """Async streaming generation — yields tokens as they arrive."""
        if not chunks:
            yield "No relevant context found to answer your question."
            return

        context = self.build_context(chunks)
        prompt_text = RAG_PROMPT.format(context=context, question=query)

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json={"model": self.model, "prompt": prompt_text, "stream": True},
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            import json
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done"):
                                break
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise OllamaConnectionError(f"Streaming failed: {e}")
