"""
Shared state for the Agentic RAG LangGraph workflow.

Difference from normal RAG
--------------------------
Normal RAG typically keeps query/docs/answer as local variables inside one
function and discards them after the answer is returned.

Agentic RAG passes a single TypedDict through every agent node so each agent
can read prior results, write its own outputs, and set `next_action` for routing.
Fields like draft_answer / verified_answer / is_grounded / retry_count do not
exist in a classic one-shot RAG pipeline.
"""

from typing import List, Optional, TypedDict


class AgenticRAGState(TypedDict):
    """Shared state passed and updated by all agents (vs ephemeral locals in normal RAG)."""

    query: str
    # Populated by retrieval agent (same idea as normal RAG's retrieved chunks)
    retrieved_docs: Optional[List[str]]
    retrieved_metadatas: Optional[List[dict]]
    # Intermediate draft — normal RAG usually has no separate draft stage
    draft_answer: Optional[str]
    # Final answer after verification — normal RAG returns the first LLM output
    verified_answer: Optional[str]
    verification_notes: Optional[str]
    is_grounded: Optional[bool]
    # Explicit error channel + routing flag — uncommon in normal RAG
    errors: List[str]
    next_action: str
    retry_count: int
