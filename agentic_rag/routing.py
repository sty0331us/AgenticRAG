"""Pure routing helpers for the LangGraph workflow (no heavy imports)."""

from __future__ import annotations

from agentic_rag.state import AgenticRAGState

ROUTE_MAP = {
    "retrieve": "retrieval",
    "reason": "reasoning",
    "verify": "verification",
    "error": "error_handler",
    "complete": "END",
}


def route_next_step(state: AgenticRAGState) -> str:
    """Resolve the next graph node from ``state['next_action']``."""
    return ROUTE_MAP.get(state.get("next_action", "retrieve"), "END")
