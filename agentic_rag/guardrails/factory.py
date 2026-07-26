"""Factory that selects the active guardrail backend from settings."""

from __future__ import annotations

from functools import lru_cache

from agentic_rag.config import Settings, get_settings
from agentic_rag.exceptions import ConfigurationError
from agentic_rag.guardrails.base import GuardrailProvider
from agentic_rag.guardrails.local import LocalGuardrailProvider
from agentic_rag.logging_config import get_logger

logger = get_logger(__name__)


def build_guardrail_provider(settings: Settings | None = None) -> GuardrailProvider:
    """
    Construct the configured guardrail provider.

    Supported GUARDRAIL_BACKEND values:
      local      — built-in OSS-style scanners (default)
      llm-guard  — ProtectAI llm-guard (optional pip install)
      bedrock    — AWS Bedrock Guardrails (optional boto3)
      azure      — Azure AI Content Safety (optional azure-ai-contentsafety)
    """
    cfg = settings or get_settings()
    backend = (cfg.guardrail_backend or "local").strip().lower()

    if backend == "local":
        return LocalGuardrailProvider(
            max_input_chars=cfg.guardrail_max_input_chars,
            max_output_chars=cfg.guardrail_max_output_chars,
        )

    if backend in {"llm-guard", "llm_guard", "llmguard"}:
        from agentic_rag.guardrails.llm_guard import LlmGuardProvider

        return LlmGuardProvider()

    if backend == "bedrock":
        from agentic_rag.guardrails.cloud import BedrockGuardrailProvider

        return BedrockGuardrailProvider(
            guardrail_id=cfg.guardrail_bedrock_id or "",
            guardrail_version=cfg.guardrail_bedrock_version,
            region_name=cfg.aws_region,
        )

    if backend == "azure":
        from agentic_rag.guardrails.cloud import AzureContentSafetyProvider

        return AzureContentSafetyProvider(
            endpoint=cfg.azure_content_safety_endpoint or "",
            api_key=cfg.azure_content_safety_key or "",
        )

    raise ConfigurationError(
        f"Unsupported GUARDRAIL_BACKEND={backend!r}. "
        "Use one of: local, llm-guard, bedrock, azure."
    )


@lru_cache(maxsize=1)
def get_guardrail_provider() -> GuardrailProvider:
    """Process-wide cached provider instance."""
    provider = build_guardrail_provider()
    logger.info("Active guardrail backend=%s", provider.name)
    return provider
