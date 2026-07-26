"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_rag.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Runtime configuration for the Agentic RAG pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: Optional[str] = Field(
        default=None,
        description="API key for Groq inference (required for query execution)",
    )
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Chat model identifier",
    )

    chroma_persist_dir: Path = Field(default=Path("./chroma_db"))
    chroma_collection: str = Field(default="agentic_rag_knowledge")

    retrieval_top_k: int = Field(default=4, ge=1, le=20)
    max_verification_retries: int = Field(default=2, ge=0, le=5)
    reasoning_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    deterministic_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    log_level: str = Field(default="INFO")

    @field_validator("groq_api_key")
    @classmethod
    def _normalize_api_key(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned or cleaned.startswith("<"):
            return None
        return cleaned

    def require_api_key(self) -> str:
        """Return the API key or raise a configuration error."""
        if not self.groq_api_key:
            raise ConfigurationError(
                "GROQ_API_KEY is not configured. "
                "Set the environment variable or add it to the .env file."
            )
        return self.groq_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached settings instance."""
    return Settings()
