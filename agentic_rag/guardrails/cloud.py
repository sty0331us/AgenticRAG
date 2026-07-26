"""
Cloud-managed guardrail providers for enterprise scale.

Why cloud backends matter for hiring / architecture discussions
---------------------------------------------------------------
Open-source scanners (local / llm-guard) are excellent for development and
cost control. At production scale, organizations typically add managed services:

  * AWS Bedrock Guardrails
      - Centralized policy (topics, denied words, PII, contextual grounding)
      - API: bedrock-runtime ApplyGuardrail
      - Scales with AWS regions / IAM / CloudWatch auditing

  * Azure AI Content Safety
      - Hate / sexual / violence / self-harm classification
      - Prompt Shields for jailbreak / indirect attacks
      - Integrates with Azure OpenAI and enterprise compliance controls

This module implements thin adapters so the LangGraph pipeline can switch from
local OSS checks to AWS or Azure without rewriting agent nodes — demonstrating
a cloud-ready, multi-provider safety architecture.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentic_rag.exceptions import ConfigurationError
from agentic_rag.guardrails.base import GuardrailDecision, GuardrailProvider, GuardrailStage
from agentic_rag.logging_config import get_logger

logger = get_logger(__name__)


class BedrockGuardrailProvider(GuardrailProvider):
    """
    AWS Bedrock Guardrails adapter.

    Requires:
      - pip install boto3
      - AWS credentials (env, shared config, or IAM role)
      - GUARDRAIL_BEDROCK_ID and optionally GUARDRAIL_BEDROCK_VERSION
    """

    name = "bedrock"

    def __init__(
        self,
        guardrail_id: str,
        guardrail_version: str = "DRAFT",
        region_name: Optional[str] = None,
    ) -> None:
        if not guardrail_id:
            raise ConfigurationError(
                "GUARDRAIL_BACKEND=bedrock requires GUARDRAIL_BEDROCK_ID "
                "(Bedrock Guardrail identifier)."
            )
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "AWS Bedrock guardrails require boto3. Install with: pip install boto3"
            ) from exc

        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self._client = boto3.client("bedrock-runtime", region_name=region_name)
        logger.info(
            "Initialized AWS Bedrock Guardrails provider id=%s version=%s",
            guardrail_id,
            guardrail_version,
        )

    def check_input(self, text: str) -> GuardrailDecision:
        return self._apply(text, stage=GuardrailStage.INPUT, source="INPUT")

    def check_output(self, text: str) -> GuardrailDecision:
        return self._apply(text, stage=GuardrailStage.OUTPUT, source="OUTPUT")

    def _apply(self, text: str, *, stage: GuardrailStage, source: str) -> GuardrailDecision:
        # ApplyGuardrail evaluates content against the configured Bedrock policy.
        response: Dict[str, Any] = self._client.apply_guardrail(
            guardrailIdentifier=self.guardrail_id,
            guardrailVersion=self.guardrail_version,
            source=source,
            content=[{"text": {"text": text}}],
        )
        action = (response.get("action") or "").upper()
        allowed = action in {"NONE", "NONE_ACTION", ""}
        # Bedrock may return assessments detailing topic/PII/word policy hits.
        reasons: List[str] = []
        if not allowed:
            reasons.append(f"bedrock_action:{action or 'UNKNOWN'}")
            assessments = response.get("assessments") or []
            for item in assessments:
                reasons.append(f"assessment:{item}")
        return GuardrailDecision(
            stage=stage,
            allowed=allowed,
            provider=self.name,
            reasons=reasons,
            sanitized_text=text if allowed else None,
            raw={"action": action},
        )


class AzureContentSafetyProvider(GuardrailProvider):
    """
    Azure AI Content Safety adapter.

    Requires:
      - pip install azure-ai-contentsafety
      - AZURE_CONTENT_SAFETY_ENDPOINT
      - AZURE_CONTENT_SAFETY_KEY
    """

    name = "azure"

    def __init__(self, endpoint: str, api_key: str) -> None:
        if not endpoint or not api_key:
            raise ConfigurationError(
                "GUARDRAIL_BACKEND=azure requires AZURE_CONTENT_SAFETY_ENDPOINT "
                "and AZURE_CONTENT_SAFETY_KEY."
            )
        try:
            from azure.ai.contentsafety import ContentSafetyClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError as exc:  # pragma: no cover
            raise ConfigurationError(
                "Azure Content Safety requires azure-ai-contentsafety. "
                "Install with: pip install azure-ai-contentsafety"
            ) from exc

        self._client = ContentSafetyClient(endpoint, AzureKeyCredential(api_key))
        logger.info("Initialized Azure AI Content Safety provider")

    def check_input(self, text: str) -> GuardrailDecision:
        return self._analyze(text, stage=GuardrailStage.INPUT)

    def check_output(self, text: str) -> GuardrailDecision:
        return self._analyze(text, stage=GuardrailStage.OUTPUT)

    def _analyze(self, text: str, *, stage: GuardrailStage) -> GuardrailDecision:
        from azure.ai.contentsafety.models import AnalyzeTextOptions

        # Severity scale is 0–6; threshold 2 blocks moderate+ categories.
        result = self._client.analyze_text(AnalyzeTextOptions(text=text))
        reasons: List[str] = []
        for cat in result.categories_analysis or []:
            severity = getattr(cat, "severity", 0) or 0
            name = getattr(cat, "category", "unknown")
            if severity >= 2:
                reasons.append(f"azure_{name}:severity={severity}")
        allowed = not reasons
        return GuardrailDecision(
            stage=stage,
            allowed=allowed,
            provider=self.name,
            reasons=reasons,
            sanitized_text=text if allowed else None,
            raw={"categories": [r for r in reasons]},
        )
