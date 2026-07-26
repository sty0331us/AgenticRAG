"""Shared state for the Agentic RAG LangGraph workflow."""

from typing import List, Optional, TypedDict


class AgenticRAGState(TypedDict):
    """Shared state passed and updated by all agents."""

    query: str
    retrieved_docs: Optional[List[str]]
    retrieved_metadatas: Optional[List[dict]]
    draft_answer: Optional[str]
    verified_answer: Optional[str]
    verification_notes: Optional[str]
    is_grounded: Optional[bool]
    errors: List[str]
    next_action: str
    retry_count: int
