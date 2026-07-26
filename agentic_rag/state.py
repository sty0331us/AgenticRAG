"""Shared LangGraph state contracts for the multi-agent pipeline."""

from __future__ import annotations

from typing import List, Optional

from typing_extensions import TypedDict


class ReActStep(TypedDict):
    """Single Thought → Action → Observation cycle recorded by an agent."""

    agent: str
    thought: str
    action: str
    observation: str


class AgenticRAGState(TypedDict):
    """
    Mutable workflow state passed between graph nodes.

    Unlike classic RAG (ephemeral locals in a single function), this state is the
    integration contract across retrieval, reasoning, verification, and error handling.
    """

    query: str
    search_query: Optional[str]
    retrieved_docs: Optional[List[str]]
    retrieved_metadatas: Optional[List[dict]]
    reasoning_thought: Optional[str]
    draft_answer: Optional[str]
    verification_thought: Optional[str]
    verified_answer: Optional[str]
    verification_notes: Optional[str]
    is_grounded: Optional[bool]
    # Guardrail decisions (input before retrieval, output before return)
    input_guardrail: Optional[dict]
    output_guardrail: Optional[dict]
    react_trace: List[ReActStep]
    errors: List[str]
    next_action: str
    retry_count: int
