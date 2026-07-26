"""Multi-agent nodes for Agentic RAG: retrieval, reasoning, verification, error handling."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_rag.llm import get_llm
from agentic_rag.state import AgenticRAGState
from agentic_rag.vectorstore import similarity_search

MAX_RETRIES = 2


def retrieval_agent(state: AgenticRAGState) -> AgenticRAGState:
    """Retrieval agent: fetch relevant knowledge from ChromaDB."""
    try:
        docs, metas = similarity_search(state["query"], k=4)
        state["retrieved_docs"] = docs
        state["retrieved_metadatas"] = metas

        if not docs:
            state["errors"] = state.get("errors", []) + [
                "No documents retrieved from ChromaDB. Run ingest first."
            ]
            state["next_action"] = "error"
        else:
            state["next_action"] = "reason"
    except Exception as exc:  # noqa: BLE001 — surface to error agent
        state["errors"] = state.get("errors", []) + [f"Retrieval failed: {exc}"]
        state["next_action"] = "error"
    return state


def reasoning_agent(state: AgenticRAGState) -> AgenticRAGState:
    """Reasoning agent: infer an answer grounded in retrieved context."""
    try:
        docs = state.get("retrieved_docs") or []
        context = "\n\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(docs))

        llm = get_llm(temperature=0.2)
        messages = [
            SystemMessage(
                content=(
                    "You are the Reasoning agent in an Agentic RAG system. "
                    "Answer the user's question using ONLY the provided retrieved context. "
                    "Be precise and cite chunk numbers like [1], [2] when relevant. "
                    "If the context is insufficient, say what is missing."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{state['query']}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    "Draft a grounded answer."
                )
            ),
        ]
        response = llm.invoke(messages)
        state["draft_answer"] = response.content
        state["next_action"] = "verify"
    except Exception as exc:  # noqa: BLE001
        state["errors"] = state.get("errors", []) + [f"Reasoning failed: {exc}"]
        state["next_action"] = "error"
    return state


def verification_agent(state: AgenticRAGState) -> AgenticRAGState:
    """Verification agent: check draft answer for accuracy and consistency vs context."""
    try:
        docs = state.get("retrieved_docs") or []
        context = "\n\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(docs))
        draft = state.get("draft_answer") or ""

        llm = get_llm(temperature=0.0)
        messages = [
            SystemMessage(
                content=(
                    "You are the Verification agent in an Agentic RAG system. "
                    "Check whether the draft answer is accurate, consistent, and grounded "
                    "in the retrieved context. Respond in this exact format:\n"
                    "GROUNDED: yes|no\n"
                    "NOTES: <brief notes>\n"
                    "FINAL_ANSWER: <corrected or confirmed answer>"
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{state['query']}\n\n"
                    f"Retrieved context:\n{context}\n\n"
                    f"Draft answer:\n{draft}"
                )
            ),
        ]
        response = llm.invoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)

        grounded = False
        notes = content
        final_answer = draft

        for line in content.splitlines():
            upper = line.strip()
            lower = upper.lower()
            if lower.startswith("grounded:"):
                grounded = "yes" in lower.split(":", 1)[1]
            elif lower.startswith("notes:"):
                notes = upper.split(":", 1)[1].strip()

        if "FINAL_ANSWER:" in content:
            final_answer = content.split("FINAL_ANSWER:", 1)[1].strip()
        elif "final_answer:" in content.lower():
            idx = content.lower().index("final_answer:")
            final_answer = content[idx + len("final_answer:") :].strip()

        state["is_grounded"] = grounded
        state["verification_notes"] = notes
        state["verified_answer"] = final_answer

        if grounded:
            state["next_action"] = "complete"
        elif state.get("retry_count", 0) < MAX_RETRIES:
            state["retry_count"] = state.get("retry_count", 0) + 1
            state["errors"] = state.get("errors", []) + [
                f"Verification failed grounding check (retry {state['retry_count']})."
            ]
            state["next_action"] = "retrieve"  # re-retrieve / re-reason pipeline
        else:
            state["errors"] = state.get("errors", []) + [
                "Answer not grounded after max retries; returning best-effort verified answer."
            ]
            state["next_action"] = "complete"
    except Exception as exc:  # noqa: BLE001
        state["errors"] = state.get("errors", []) + [f"Verification failed: {exc}"]
        state["next_action"] = "error"
    return state


def error_handler_agent(state: AgenticRAGState) -> AgenticRAGState:
    """Error agent: fallback, summarize errors, and terminate safely."""
    errors = state.get("errors") or []
    error_summary = "; ".join(errors) if errors else "Unknown error"
    state["verified_answer"] = (
        state.get("draft_answer")
        or f"Unable to complete Agentic RAG for this query. Errors: {error_summary}"
    )
    state["verification_notes"] = f"Handled by error agent. {error_summary}"
    state["is_grounded"] = False
    state["next_action"] = "complete"
    return state
