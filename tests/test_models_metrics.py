"""Unit tests for PipelineResult mapping and metrics helpers."""

from __future__ import annotations

from agentic_rag.metrics import add_metric, record_metric, timed_section
from agentic_rag.models import PipelineResult


def test_pipeline_result_from_state_builds_sources() -> None:
    state = {
        "query": "What is Agentic RAG?",
        "search_query": "agentic rag agents",
        "retrieved_docs": ["evidence"],
        "retrieved_metadatas": [{"topic": "rag"}],
        "retrieved_scores": [0.88],
        "reasoning_thought": "map evidence",
        "draft_answer": "draft",
        "verification_thought": "check claims",
        "verified_answer": "final",
        "verification_notes": "ok",
        "is_grounded": True,
        "react_trace": [
            {
                "agent": "retrieval",
                "thought": "t",
                "action": "a",
                "observation": "o",
            }
        ],
        "errors": [],
        "next_action": "complete",
        "retry_count": 1,
        "metrics": {
            "total_seconds": 1.25,
            "retrieval_seconds": 0.4,
            "reasoning_seconds": 0.5,
            "verification_seconds": 0.35,
        },
    }
    result = PipelineResult.from_state(state)  # type: ignore[arg-type]
    assert result.answer == "final"
    assert result.is_grounded is True
    assert result.retry_count == 1
    assert len(result.sources) == 1
    assert result.sources[0].citation_label == "[1]"
    assert result.sources[0].score == 0.88
    assert result.sources[0].metadata["topic"] == "rag"
    assert result.metrics.total_seconds == 1.25
    assert result.metrics.retrieval_seconds == 0.4
    dumped = result.to_dict()
    assert "sources" in dumped
    assert "metrics" in dumped


def test_metric_helpers_accumulate() -> None:
    state: dict = {"metrics": {}}
    record_metric(state, "total_seconds", 1.0)  # type: ignore[arg-type]
    add_metric(state, "retrieval_seconds", 0.2)  # type: ignore[arg-type]
    add_metric(state, "retrieval_seconds", 0.3)  # type: ignore[arg-type]
    with timed_section(state, "reasoning_seconds"):  # type: ignore[arg-type]
        pass
    assert state["metrics"]["total_seconds"] == 1.0
    assert state["metrics"]["retrieval_seconds"] == 0.5
    assert state["metrics"]["reasoning_seconds"] >= 0.0
