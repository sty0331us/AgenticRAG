"""Domain-specific exceptions for the Agentic RAG pipeline."""

from __future__ import annotations


class AgenticRAGError(Exception):
    """Base exception for pipeline failures."""


class ConfigurationError(AgenticRAGError):
    """Raised when required configuration is missing or invalid."""


class RetrievalError(AgenticRAGError):
    """Raised when vector retrieval cannot complete successfully."""


class EmptyKnowledgeBaseError(RetrievalError):
    """Raised when the vector store contains no documents."""


class LLMInvocationError(AgenticRAGError):
    """Raised when an LLM call fails."""


class GuardrailBlockedError(AgenticRAGError):
    """Raised when content is blocked by a configured guardrail policy."""
