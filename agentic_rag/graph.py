"""
LangGraph workflow compilation and pipeline service facade.

Classic RAG executes a fixed retrieve → generate sequence. This module compiles
a conditional StateGraph in which each node is a ReAct agent and routing is
driven by shared state (`next_action`), including verification-triggered retries
and input/output guardrail gates.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic_rag.agents import (
    error_handler_agent,
    reasoning_agent,
    retrieval_agent,
    verification_agent,
)
from agentic_rag.config import Settings, get_settings
from agentic_rag.guardrails.nodes import input_guardrail_agent, output_guardrail_agent
from agentic_rag.logging_config import get_logger
from agentic_rag.models import PipelineResult
from agentic_rag.state import AgenticRAGState

logger = get_logger(__name__)

_ROUTE_MAP = {
    "retrieve": "retrieval",
    "reason": "reasoning",
    "verify": "verification",
    # After successful verification, enforce output safety before returning.
    "output_guard": "output_guardrail",
    "error": "error_handler",
    "complete": "END",
}


def route_next_step(state: AgenticRAGState) -> str:
    """Resolve the next graph node from `state['next_action']`."""
    return _ROUTE_MAP.get(state.get("next_action", "retrieve"), "END")


def build_workflow() -> CompiledStateGraph:
    """
    Construct and compile the Agentic RAG StateGraph.

    Order of safety + intelligence nodes:
      input_guardrail → retrieval → reasoning → verification → output_guardrail
    """
    workflow = StateGraph(AgenticRAGState)
    workflow.add_node("input_guardrail", input_guardrail_agent)
    workflow.add_node("retrieval", retrieval_agent)
    workflow.add_node("reasoning", reasoning_agent)
    workflow.add_node("verification", verification_agent)
    workflow.add_node("output_guardrail", output_guardrail_agent)
    workflow.add_node("error_handler", error_handler_agent)
    workflow.set_entry_point("input_guardrail")

    destinations = {
        "retrieval": "retrieval",
        "reasoning": "reasoning",
        "verification": "verification",
        "output_guardrail": "output_guardrail",
        "error_handler": "error_handler",
        "END": END,
    }
    for node in (
        "input_guardrail",
        "retrieval",
        "reasoning",
        "verification",
        "output_guardrail",
        "error_handler",
    ):
        workflow.add_conditional_edges(node, route_next_step, destinations)

    return workflow.compile()


@lru_cache(maxsize=1)
def get_compiled_workflow() -> CompiledStateGraph:
    """Return a process-wide compiled graph instance."""
    return build_workflow()


def initial_state(query: str) -> AgenticRAGState:
    """Create the initial shared state for a query run."""
    return {
        "query": query.strip(),
        "search_query": None,
        "retrieved_docs": None,
        "retrieved_metadatas": None,
        "reasoning_thought": None,
        "draft_answer": None,
        "verification_thought": None,
        "verified_answer": None,
        "verification_notes": None,
        "is_grounded": None,
        "input_guardrail": None,
        "output_guardrail": None,
        "react_trace": [],
        "errors": [],
        "next_action": "retrieve",
        "retry_count": 0,
    }


class AgenticRAGPipeline:
    """
    Production entry point for executing the multi-agent RAG workflow.

    Example:
        pipeline = AgenticRAGPipeline()
        result = pipeline.invoke("What is Agentic RAG?")
        print(result.answer)
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.settings.require_api_key()
        # Rebuild provider cache if settings object is injected for tests.
        self._app = get_compiled_workflow()

    def invoke(self, query: str) -> PipelineResult:
        """Execute guardrails + retrieval → reasoning → verification pipeline."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        logger.info(
            "Pipeline invoke started guardrail_backend=%s",
            self.settings.guardrail_backend,
        )
        final_state: AgenticRAGState = self._app.invoke(initial_state(query))
        result = PipelineResult.from_state(final_state)
        logger.info(
            "Pipeline invoke completed grounded=%s errors=%s",
            result.is_grounded,
            len(result.errors),
        )
        return result


def create_workflow() -> CompiledStateGraph:
    """Compile a fresh workflow graph (useful for tests and introspection)."""
    return build_workflow()


def run_agentic_rag(query: str) -> PipelineResult:
    """Convenience wrapper around AgenticRAGPipeline.invoke()."""
    return AgenticRAGPipeline().invoke(query)
