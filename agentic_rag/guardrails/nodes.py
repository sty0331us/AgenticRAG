"""LangGraph nodes that enforce input/output guardrails."""

from __future__ import annotations

from agentic_rag.guardrails import get_guardrail_provider
from agentic_rag.logging_config import get_logger
from agentic_rag.react import append_error, append_react_step
from agentic_rag.state import AgenticRAGState

logger = get_logger(__name__)

_BLOCKED_MESSAGE = (
    "Request blocked by safety guardrails. "
    "The submitted content violated configured safety policy."
)


def input_guardrail_agent(state: AgenticRAGState) -> AgenticRAGState:
    """
    Input guardrail node (runs before retrieval).

    Uses the configured provider (local OSS heuristics, ProtectAI llm-guard,
    AWS Bedrock Guardrails, or Azure AI Content Safety).
    """
    provider = get_guardrail_provider()
    decision = provider.check_input(state["query"])
    state["input_guardrail"] = decision.model_dump()

    append_react_step(
        state,
        agent="input_guardrail",
        thought=f"Evaluate inbound query with provider={provider.name}.",
        action=f"{provider.name}.check_input",
        observation=(
            f"allowed={decision.allowed}; reasons={decision.reasons or ['none']}"
        ),
    )

    if decision.allowed:
        state["next_action"] = "retrieve"
        logger.info("Input guardrail passed provider=%s", provider.name)
    else:
        append_error(state, f"Input guardrail blocked request: {decision.reasons}")
        state["verified_answer"] = _BLOCKED_MESSAGE
        state["is_grounded"] = False
        state["next_action"] = "error"
        logger.warning("Input guardrail blocked provider=%s reasons=%s", provider.name, decision.reasons)
    return state


def output_guardrail_agent(state: AgenticRAGState) -> AgenticRAGState:
    """
    Output guardrail node (runs after verification, before returning to callers).

    Cloud note: at scale, AWS Bedrock Guardrails / Azure Content Safety provide
    centralized policy versioning and audit trails across regions.
    """
    provider = get_guardrail_provider()
    answer = state.get("verified_answer") or state.get("draft_answer") or ""
    decision = provider.check_output(answer)
    state["output_guardrail"] = decision.model_dump()

    append_react_step(
        state,
        agent="output_guardrail",
        thought=f"Evaluate outbound answer with provider={provider.name}.",
        action=f"{provider.name}.check_output",
        observation=(
            f"allowed={decision.allowed}; reasons={decision.reasons or ['none']}"
        ),
    )

    if decision.allowed:
        if decision.sanitized_text:
            state["verified_answer"] = decision.sanitized_text
        state["next_action"] = "complete"
        logger.info("Output guardrail passed provider=%s", provider.name)
    else:
        append_error(state, f"Output guardrail blocked response: {decision.reasons}")
        state["verified_answer"] = _BLOCKED_MESSAGE
        state["is_grounded"] = False
        state["next_action"] = "error"
        logger.warning(
            "Output guardrail blocked provider=%s reasons=%s",
            provider.name,
            decision.reasons,
        )
    return state
