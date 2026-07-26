"""Agentic RAG: multi-agent retrieval, reasoning, and verification with LangGraph."""

from agentic_rag.graph import AgenticRAGPipeline, create_workflow, run_agentic_rag
from agentic_rag.models import PipelineResult

__all__ = [
    "AgenticRAGPipeline",
    "PipelineResult",
    "create_workflow",
    "run_agentic_rag",
]

__version__ = "1.0.0"
