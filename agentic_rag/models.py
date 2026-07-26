"""Typed response models for pipeline callers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agentic_rag.state import AgenticRAGState


class ReActStepModel(BaseModel):
    """Serializable ReAct audit-trail entry."""

    agent: str
    thought: str
    action: str
    observation: str


class PipelineResult(BaseModel):
    """Stable public response returned by AgenticRAGPipeline.invoke()."""

    query: str
    search_query: Optional[str] = None
    answer: str
    draft_answer: Optional[str] = None
    is_grounded: bool = False
    verification_notes: Optional[str] = None
    reasoning_thought: Optional[str] = None
    verification_thought: Optional[str] = None
    retrieved_docs: List[str] = Field(default_factory=list)
    react_trace: List[ReActStepModel] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    retry_count: int = 0
    input_guardrail: Optional[Dict[str, Any]] = None
    output_guardrail: Optional[Dict[str, Any]] = None

    @classmethod
    def from_state(cls, state: AgenticRAGState) -> "PipelineResult":
        """Map LangGraph state into the public response model."""
        return cls(
            query=state["query"],
            search_query=state.get("search_query"),
            answer=state.get("verified_answer") or "",
            draft_answer=state.get("draft_answer"),
            is_grounded=bool(state.get("is_grounded")),
            verification_notes=state.get("verification_notes"),
            reasoning_thought=state.get("reasoning_thought"),
            verification_thought=state.get("verification_thought"),
            retrieved_docs=list(state.get("retrieved_docs") or []),
            react_trace=[ReActStepModel(**step) for step in (state.get("react_trace") or [])],
            errors=list(state.get("errors") or []),
            retry_count=int(state.get("retry_count") or 0),
            input_guardrail=state.get("input_guardrail"),
            output_guardrail=state.get("output_guardrail"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON output."""
        return self.model_dump()
