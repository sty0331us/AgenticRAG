"""
Pluggable safety guardrails for input and output validation.

Default backend: local heuristic + regex scanners (no external dependency).

Optional open-source backend:
  - ProtectAI ``llm-guard`` (Apache-2.0) — widely used for prompt-injection /
    toxicity / PII scanning. Enable with GUARDRAIL_BACKEND=llm-guard.

Cloud backends (scalability / managed policy at enterprise scale):
  - AWS Bedrock Guardrails  — GUARDRAIL_BACKEND=bedrock
  - Azure AI Content Safety — GUARDRAIL_BACKEND=azure

The provider interface is intentional: open-source scanners are suitable for
local development and cost control; AWS / Azure managed guardrails are the
preferred path when scaling to multi-region production workloads, centralized
policy management, and compliance auditing.
"""

from __future__ import annotations

from agentic_rag.guardrails.base import GuardrailDecision, GuardrailProvider
from agentic_rag.guardrails.factory import get_guardrail_provider

__all__ = [
    "GuardrailDecision",
    "GuardrailProvider",
    "get_guardrail_provider",
]
