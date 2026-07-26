"""
Shared state for the Agentic RAG LangGraph workflow.

Difference from normal RAG
--------------------------
Normal RAG keeps query/docs/answer as locals in one function.

Agentic RAG shares a TypedDict across agents and records an explicit ReAct
trace (Thought → Action → Observation) so you can inspect how each agent
reasoned and what it did — not just the final string.
"""

from typing import List, Optional, TypedDict


class ReActStep(TypedDict):
    """One Thought → Action → Observation cycle from a single agent."""

    agent: str
    thought: str
    action: str
    observation: str


class AgenticRAGState(TypedDict):
    """Shared state passed and updated by all agents."""

    query: str
    # Retrieval agent may rewrite the query after its Thought step
    search_query: Optional[str]
    retrieved_docs: Optional[List[str]]
    retrieved_metadatas: Optional[List[dict]]
    # Reasoning agent: explicit plan + draft (Think then Act)
    reasoning_thought: Optional[str]
    draft_answer: Optional[str]
    # Verification agent outputs
    verification_thought: Optional[str]
    verified_answer: Optional[str]
    verification_notes: Optional[str]
    is_grounded: Optional[bool]
    # Full ReAct audit trail across the graph
    react_trace: List[ReActStep]
    errors: List[str]
    next_action: str
    retry_count: int
