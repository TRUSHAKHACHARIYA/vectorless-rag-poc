from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_temperature: float = 0.1
    ollama_num_ctx: int = 4096

    bm25_top_k: int = 5
    bm25_chunk_size: int = 500
    bm25_chunk_overlap: int = 50

    max_file_size_mb: int = 10
    allowed_extensions: str = "pdf,txt,md,docx"
    upload_dir: str = "data/uploads"

    @property
    def allowed_ext_list(self) -> list[str]:
        return [ext.strip() for ext in self.allowed_extensions.split(",") if ext.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()
