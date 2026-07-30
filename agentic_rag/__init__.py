"""Agentic RAG: multi-agent retrieval, reasoning, and verification with LangGraph."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AgenticRAGPipeline",
    "PipelineMetrics",
    "PipelineResult",
    "RetrievedSource",
    "create_workflow",
    "run_agentic_rag",
]

__version__ = "1.0.0"


def __getattr__(name: str) -> Any:
    """Lazy-load public exports so lightweight modules can be imported in isolation."""
    if name in {"AgenticRAGPipeline", "create_workflow", "run_agentic_rag"}:
        from agentic_rag.graph import AgenticRAGPipeline, create_workflow, run_agentic_rag

        exports = {
            "AgenticRAGPipeline": AgenticRAGPipeline,
            "create_workflow": create_workflow,
            "run_agentic_rag": run_agentic_rag,
        }
        return exports[name]
    if name in {"PipelineMetrics", "PipelineResult", "RetrievedSource"}:
        from agentic_rag.models import PipelineMetrics, PipelineResult, RetrievedSource

        exports = {
            "PipelineMetrics": PipelineMetrics,
            "PipelineResult": PipelineResult,
            "RetrievedSource": RetrievedSource,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
