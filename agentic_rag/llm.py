"""
LLM client factory for agent inference.

Multiple agents share the same provider with role-specific temperatures.
Classic RAG typically invokes the model once after retrieval; this package
invokes it per specialized agent (query planning, drafting, verification).
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_groq import ChatGroq

from agentic_rag.config import Settings, get_settings
from agentic_rag.exceptions import ConfigurationError, LLMInvocationError
from agentic_rag.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=8)
def _build_llm(model: str, api_key: str, temperature: float) -> ChatGroq:
    return ChatGroq(model=model, api_key=api_key, temperature=temperature)


def get_llm(
    temperature: float | None = None,
    settings: Settings | None = None,
) -> ChatGroq:
    """
    Return a configured chat model client.

    Raises:
        ConfigurationError: When the API key is not configured.
    """
    try:
        cfg = settings or get_settings()
        api_key = cfg.require_api_key()
    except ConfigurationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ConfigurationError(str(exc)) from exc

    temp = cfg.deterministic_temperature if temperature is None else temperature
    logger.debug("Creating LLM client model=%s temperature=%s", cfg.groq_model, temp)
    return _build_llm(cfg.groq_model, api_key, temp)


def message_text(content: object) -> str:
    """Normalize LangChain message content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def invoke_text(llm: ChatGroq, messages: list) -> str:
    """Invoke the LLM and return normalized text, wrapping provider errors."""
    try:
        response = llm.invoke(messages)
    except Exception as exc:  # noqa: BLE001
        raise LLMInvocationError(f"LLM invocation failed: {exc}") from exc
    return message_text(response.content)
