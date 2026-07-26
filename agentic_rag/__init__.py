"""
Agentic RAG package: multi-agent retrieval, reasoning, and verification.

Difference from normal RAG
--------------------------
Normal RAG is usually one linear script: retrieve → single LLM answer.
Agentic RAG splits work across specialized agents orchestrated by LangGraph,
with shared state, dynamic routing, verification, and retries.
"""

from agentic_rag.graph import create_workflow, run_agentic_rag

__all__ = ["create_workflow", "run_agentic_rag"]
