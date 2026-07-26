"""
LangGraph workflow: Retrieval → Reasoning → Verification (+ error handling).

Difference from normal RAG
--------------------------
Normal RAG control flow is a fixed script (no graph, no router):
  docs = retrieve(query)
  answer = llm(query, docs)
  return answer

Agentic RAG builds a StateGraph:
  - nodes = specialized agents
  - conditional edges = route_next_step(state)
  - shared state carries intermediates (draft, grounding, retries)

That enables loops (re-retrieve after failed verification) and error detours,
which a classic linear RAG path cannot express without ad-hoc if/else glue.
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
    """
    Dynamic routing based on shared state next_action.

    vs normal RAG: there is usually no router — the next step is hard-coded.
    """
    routing = {
        "retrieve": "retrieval",
        "reason": "reasoning",
        "verify": "verification",
        "error": "error_handler",
        "complete": "END",
    }
    return routing.get(state.get("next_action", "retrieve"), "END")


def create_workflow():
    """
    Build and compile the Agentic RAG StateGraph.

    vs normal RAG: this graph IS the pipeline. Normal RAG would not register
    nodes/edges or compile a runnable graph object.
    """
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

    # Conditional edges after every node — key Agentic difference from linear RAG
    for node in ("retrieval", "reasoning", "verification", "error_handler"):
        workflow.add_conditional_edges(node, route_next_step, route_map)

    return workflow.compile()


def run_agentic_rag(query: str) -> AgenticRAGState:
    """
    Run the full Agentic RAG pipeline for a user query.

    vs normal RAG: returns rich state (draft, verified answer, grounding, errors),
    not only a final string.
    """
    app = create_workflow()
    initial_state: AgenticRAGState = {
        "query": query,
        "retrieved_docs": None,
        "retrieved_metadatas": None,
        "draft_answer": None,
        "verified_answer": None,
        "verification_notes": None,
        "is_grounded": None,
        "errors": [],
        "next_action": "retrieve",
        "retry_count": 0,
    }
    return app.invoke(initial_state)
