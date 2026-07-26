"""
Optional ProtectAI llm-guard integration (popular open-source guardrail library).

Package: https://github.com/protectai/llm-guard  (Apache-2.0)

Install:
    pip install llm-guard

Enable:
    GUARDRAIL_BACKEND=llm-guard

llm-guard provides model-backed scanners for prompt injection, toxicity, secrets,
and PII. Prefer this over pure regex when offline ML inference is acceptable.

Alternatives in the same OSS category (not wired here, documented for evaluation):
  - Guardrails AI (guardrails-ai) — composable validators / Hub
  - NVIDIA NeMo Guardrails — programmable conversational rails (Colang)
"""

from __future__ import annotations

from agentic_rag.exceptions import ConfigurationError
from agentic_rag.guardrails.base import GuardrailDecision, GuardrailProvider, GuardrailStage
from agentic_rag.logging_config import get_logger

logger = get_logger(__name__)


class LlmGuardProvider(GuardrailProvider):
    """Adapter around ProtectAI llm-guard input/output scanners."""

    name = "llm-guard"

    def __init__(self) -> None:
        try:
            from llm_guard.input_scanners import PromptInjection, Toxicity
            from llm_guard.output_scanners import Sensitive
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ConfigurationError(
                "GUARDRAIL_BACKEND=llm-guard requires the optional package "
                "`llm-guard`. Install with: pip install llm-guard"
            ) from exc

        # Thresholds are conservative defaults; tune per risk appetite.
        self._input_scanners = [
            PromptInjection(threshold=0.75),
            Toxicity(threshold=0.7),
        ]
        self._output_scanners = [
            Sensitive(entity_types=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD"]),
        ]
        logger.info("Initialized ProtectAI llm-guard provider")

    def check_input(self, text: str) -> GuardrailDecision:
        reasons = []
        sanitized = text
        for scanner in self._input_scanners:
            sanitized, is_valid, risk_score = scanner.scan(sanitized)
            if not is_valid:
                reasons.append(f"{scanner.__class__.__name__}:risk={risk_score}")
        allowed = not reasons
        return GuardrailDecision(
            stage=GuardrailStage.INPUT,
            allowed=allowed,
            provider=self.name,
            reasons=reasons,
            sanitized_text=sanitized if allowed else None,
            raw={"scanner_count": len(self._input_scanners)},
        )

    def check_output(self, text: str) -> GuardrailDecision:
        reasons = []
        sanitized = text
        for scanner in self._output_scanners:
            sanitized, is_valid, risk_score = scanner.scan(sanitized)
            if not is_valid:
                reasons.append(f"{scanner.__class__.__name__}:risk={risk_score}")
        allowed = not reasons
        return GuardrailDecision(
            stage=GuardrailStage.OUTPUT,
            allowed=allowed,
            provider=self.name,
            reasons=reasons,
            sanitized_text=sanitized if allowed else None,
            raw={"scanner_count": len(self._output_scanners)},
        )
