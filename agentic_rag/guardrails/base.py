"""Guardrail contracts shared by local and cloud providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class GuardrailStage(str, Enum):
    """Where in the pipeline the check runs."""

    INPUT = "input"
    OUTPUT = "output"


class GuardrailDecision(BaseModel):
    """Normalized result from any guardrail backend."""

    stage: GuardrailStage
    allowed: bool
    provider: str
    reasons: List[str] = Field(default_factory=list)
    sanitized_text: Optional[str] = None
    raw: Optional[dict] = None


class GuardrailProvider(ABC):
    """
    Abstract guardrail backend.

    Implementations may call open-source libraries (e.g. ProtectAI llm-guard)
    or managed cloud APIs (AWS Bedrock Guardrails, Azure AI Content Safety).
    """

    name: str = "abstract"

    @abstractmethod
    def check_input(self, text: str) -> GuardrailDecision:
        """Validate user / query text before retrieval and generation."""

    @abstractmethod
    def check_output(self, text: str) -> GuardrailDecision:
        """Validate model output before it is returned to callers."""
