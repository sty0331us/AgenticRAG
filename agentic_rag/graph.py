"""
LangGraph workflow: Retrieval → Reasoning → Verification (+ error handling).

Difference from normal RAG
--------------------------
Normal RAG is a fixed script. Agentic RAG is a StateGraph where each node is a
ReAct agent (Thought → Action → Observation) and edges are chosen dynamically
via next_action (including verification-driven retries).
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agentic_rag.agents import (
    error_handler_agent,
    reasoning_agent,
    retrieval_agent,
    verification_agent,
)
from agentic_rag.state import AgenticRAGState


def route_next_step(state: AgenticRAGState) -> str:
    """Dynamic routing from each agent's next_action (vs hard-coded normal RAG)."""
    routing = {
        "retrieve": "retrieval",
        "reason": "reasoning",
        "verify": "verification",
        "error": "error_handler",
        "complete": "END",
    }
    return routing.get(state.get("next_action", "retrieve"), "END")


def create_workflow():
    """Compile the Agentic RAG graph of ReAct agents."""
    workflow = StateGraph(AgenticRAGState)

    workflow.add_node("retrieval", retrieval_agent)
    workflow.add_node("reasoning", reasoning_agent)
    workflow.add_node("verification", verification_agent)
    workflow.add_node("error_handler", error_handler_agent)

    workflow.set_entry_point("retrieval")

    route_map = {
        "retrieval": "retrieval",
        "reasoning": "reasoning",
        "verification": "verification",
        "error_handler": "error_handler",
        "END": END,
    }

    for node in ("retrieval", "reasoning", "verification", "error_handler"):
        workflow.add_conditional_edges(node, route_next_step, route_map)

    return workflow.compile()


def run_agentic_rag(query: str) -> AgenticRAGState:
    """Run the full multi-agent ReAct pipeline; returns rich state + react_trace."""
    app = create_workflow()
    initial_state: AgenticRAGState = {
        "query": query,
        "search_query": None,
        "retrieved_docs": None,
        "retrieved_metadatas": None,
        "reasoning_thought": None,
        "draft_answer": None,
        "verification_thought": None,
        "verified_answer": None,
        "verification_notes": None,
        "is_grounded": None,
        "react_trace": [],
        "errors": [],
        "next_action": "retrieve",
        "retry_count": 0,
    }
    return app.invoke(initial_state)
