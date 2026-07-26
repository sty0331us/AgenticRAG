"""
Local open-source-style guardrail scanners (default backend).

These checks run fully on-premise with zero cloud cost. They approximate the
categories covered by popular OSS stacks such as ProtectAI llm-guard and
Guardrails AI validators (prompt injection, PII leakage, toxic content).

For production scale, swap to GUARDRAIL_BACKEND=llm-guard, bedrock, or azure
via the factory — the LangGraph nodes do not need to change.
"""

from __future__ import annotations

import re
from typing import List, Pattern, Tuple

from agentic_rag.guardrails.base import GuardrailDecision, GuardrailProvider, GuardrailStage
from agentic_rag.logging_config import get_logger

logger = get_logger(__name__)

# Prompt-injection / jailbreak heuristics commonly blocked by OSS scanners.
_INJECTION_PATTERNS: Tuple[Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"disregard\s+(all\s+)?(previous|prior)\s+(rules|instructions)",
        r"system\s+prompt",
        r"jailbreak",
        r"dan\s+mode",
        r"developer\s+mode\s+enabled",
        r"bypass\s+(safety|guardrail|filter)",
        r"reveal\s+(your|the)\s+(hidden|system)\s+prompt",
    )
)

# Lightweight PII detectors (SSN / payment card / email). Cloud services provide
# broader coverage; this keeps a free baseline without model downloads.
_PII_PATTERNS: Tuple[Tuple[str, Pattern[str]], ...] = (
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
)

_TOXIC_TERMS = (
    "kill yourself",
    "build a bomb",
    "how to make a bomb",
    "credit card dump",
)


class LocalGuardrailProvider(GuardrailProvider):
    """Dependency-free local scanner suitable for demos and air-gapped envs."""

    name = "local"

    def __init__(self, max_input_chars: int = 4000, max_output_chars: int = 8000) -> None:
        self.max_input_chars = max_input_chars
        self.max_output_chars = max_output_chars

    def check_input(self, text: str) -> GuardrailDecision:
        reasons = self._scan_common(text)
        if len(text) > self.max_input_chars:
            reasons.append(f"input_length_exceeded:{len(text)}>{self.max_input_chars}")
        reasons.extend(self._scan_injection(text))
        allowed = not reasons
        logger.info("Local input guardrail allowed=%s reasons=%s", allowed, reasons)
        return GuardrailDecision(
            stage=GuardrailStage.INPUT,
            allowed=allowed,
            provider=self.name,
            reasons=reasons,
            sanitized_text=text if allowed else None,
        )

    def check_output(self, text: str) -> GuardrailDecision:
        reasons = self._scan_common(text)
        if len(text) > self.max_output_chars:
            reasons.append(f"output_length_exceeded:{len(text)}>{self.max_output_chars}")
        # Redact obvious PII in outputs when possible (soft mitigation).
        sanitized, pii_hits = self._redact_pii(text)
        if pii_hits:
            reasons.append(f"pii_detected:{','.join(pii_hits)}")
        # Treat PII as block for policy clarity in this reference implementation.
        allowed = not reasons
        logger.info("Local output guardrail allowed=%s reasons=%s", allowed, reasons)
        return GuardrailDecision(
            stage=GuardrailStage.OUTPUT,
            allowed=allowed,
            provider=self.name,
            reasons=reasons,
            sanitized_text=sanitized if allowed else None,
        )

    def _scan_common(self, text: str) -> List[str]:
        reasons: List[str] = []
        lowered = text.lower()
        for term in _TOXIC_TERMS:
            if term in lowered:
                reasons.append(f"toxic_content:{term}")
        return reasons

    def _scan_injection(self, text: str) -> List[str]:
        hits: List[str] = []
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                hits.append(f"prompt_injection:{pattern.pattern}")
        return hits

    def _redact_pii(self, text: str) -> Tuple[str, List[str]]:
        sanitized = text
        hits: List[str] = []
        for label, pattern in _PII_PATTERNS:
            if pattern.search(sanitized):
                hits.append(label)
                sanitized = pattern.sub(f"[REDACTED_{label.upper()}]", sanitized)
        return sanitized, hits
