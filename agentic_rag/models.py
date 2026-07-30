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


class RetrievedSource(BaseModel):
    """One retrieved evidence chunk with score and metadata for citation."""

    index: int
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def citation_label(self) -> str:
        """Stable citation token matching format_context indices, e.g. ``[1]``."""
        return f"[{self.index}]"


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
    retrieved_scores: List[float] = Field(default_factory=list)
    retrieved_metadatas: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[RetrievedSource] = Field(default_factory=list)
    react_trace: List[ReActStepModel] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    retry_count: int = 0

    @classmethod
    def from_state(cls, state: AgenticRAGState) -> "PipelineResult":
        """Map LangGraph state into the public response model."""
        docs = list(state.get("retrieved_docs") or [])
        metas = list(state.get("retrieved_metadatas") or [])
        scores = list(state.get("retrieved_scores") or [])
        while len(metas) < len(docs):
            metas.append({})
        while len(scores) < len(docs):
            scores.append(0.0)

        sources = [
            RetrievedSource(
                index=i + 1,
                text=doc,
                score=scores[i] if i < len(scores) else None,
                metadata=dict(metas[i]) if i < len(metas) else {},
            )
            for i, doc in enumerate(docs)
        ]

        return cls(
            query=state["query"],
            search_query=state.get("search_query"),
            answer=state.get("verified_answer") or "",
            draft_answer=state.get("draft_answer"),
            is_grounded=bool(state.get("is_grounded")),
            verification_notes=state.get("verification_notes"),
            reasoning_thought=state.get("reasoning_thought"),
            verification_thought=state.get("verification_thought"),
            retrieved_docs=docs,
            retrieved_scores=scores[: len(docs)],
            retrieved_metadatas=[dict(m) for m in metas[: len(docs)]],
            sources=sources,
            react_trace=[ReActStepModel(**step) for step in (state.get("react_trace") or [])],
            errors=list(state.get("errors") or []),
            retry_count=int(state.get("retry_count") or 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON output."""
        return self.model_dump()
