"""
Multi-agent nodes with explicit ReAct loops: Thought → Action → Observation.

Difference from normal RAG
--------------------------
Normal RAG: retrieve → one generate call → answer.

Agentic RAG agents each run a mini ReAct cycle:
  1. Thought  — decide what to do / how to interpret evidence
  2. Action   — call a tool or produce a structured artifact
  3. Observation — record result into shared state + react_trace

Agents
------
  retrieval_agent    — Think (rewrite query) → Act (ChromaDB search) → Observe
  reasoning_agent    — Think (plan answer from evidence) → Act (draft) → Observe
  verification_agent — Think (check claims) → Act (accept/correct) → Observe
  error_handler_agent— Act (safe fallback) when any step fails
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_rag.llm import get_llm
from agentic_rag.react import append_react_step, format_context, parse_labeled_blocks
from agentic_rag.state import AgenticRAGState
from agentic_rag.vectorstore import similarity_search

MAX_RETRIES = 2


def retrieval_agent(state: AgenticRAGState) -> AgenticRAGState:
    """
    Retrieval agent (ReAct).

    Thought  → reformulate / focus the user query for vector search
    Action   → similarity_search(search_query) against ChromaDB
    Observe  → store docs; route to reasoning or error
    """
    try:
        llm = get_llm(temperature=0.0)
        think = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the Retrieval agent in an Agentic RAG system.\n"
                        "Run one ReAct cycle focused on SEARCH PLANNING only.\n"
                        "Respond in this exact format:\n"
                        "THOUGHT: <what information is needed and why>\n"
                        "SEARCH_QUERY: <concise query optimized for vector search>"
                    )
                ),
                HumanMessage(content=f"User question:\n{state['query']}"),
            ]
        )
        think_text = think.content if isinstance(think.content, str) else str(think.content)
        parsed = parse_labeled_blocks(think_text, ["THOUGHT", "SEARCH_QUERY"])
        thought = parsed.get("THOUGHT") or "Search the knowledge base for passages relevant to the question."
        search_query = parsed.get("SEARCH_QUERY") or state["query"]
        state["search_query"] = search_query

        # ACTION: tool call (vector search) — this is the "Act" half of ReAct
        docs, metas = similarity_search(search_query, k=4)
        state["retrieved_docs"] = docs
        state["retrieved_metadatas"] = metas

        observation = (
            f"Retrieved {len(docs)} chunk(s) for search_query={search_query!r}."
            if docs
            else "No documents retrieved from ChromaDB."
        )
        append_react_step(
            state,
            agent="retrieval",
            thought=thought,
            action=f"similarity_search(query={search_query!r}, k=4)",
            observation=observation,
        )

        if not docs:
            state["errors"] = state.get("errors", []) + [
                "No documents retrieved from ChromaDB. Run ingest first."
            ]
            state["next_action"] = "error"
        else:
            state["next_action"] = "reason"
    except Exception as exc:  # noqa: BLE001
        state["errors"] = state.get("errors", []) + [f"Retrieval failed: {exc}"]
        state["next_action"] = "error"
    return state


def reasoning_agent(state: AgenticRAGState) -> AgenticRAGState:
    """
    Reasoning agent (ReAct).

    Thought  → analyze evidence, decide how to answer, note gaps
    Action   → write a grounded DRAFT_ANSWER (still not final)
    Observe  → store thought + draft; route to verification

    vs normal RAG: generate is split into explicit reasoning then drafting,
    and the draft is not trusted until verification.
    """
    try:
        docs = state.get("retrieved_docs") or []
        context = format_context(docs)

        llm = get_llm(temperature=0.2)
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the Reasoning agent in an Agentic RAG system.\n"
                        "Use ReAct: first think, then act by drafting an answer.\n"
                        "Use ONLY the retrieved context. Cite chunks like [1], [2].\n"
                        "Respond in this exact format:\n"
                        "THOUGHT: <how the evidence maps to the question; gaps if any>\n"
                        "DRAFT_ANSWER: <grounded answer draft>"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{state['query']}\n\n"
                        f"Search query used:\n{state.get('search_query') or state['query']}\n\n"
                        f"Retrieved context:\n{context}"
                    )
                ),
            ]
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = parse_labeled_blocks(content, ["THOUGHT", "DRAFT_ANSWER"])
        thought = parsed.get("THOUGHT") or content
        draft = parsed.get("DRAFT_ANSWER") or content

        state["reasoning_thought"] = thought
        state["draft_answer"] = draft

        append_react_step(
            state,
            agent="reasoning",
            thought=thought,
            action="draft_answer_from_context",
            observation=f"Draft produced ({len(draft)} chars). Awaiting verification.",
        )
        state["next_action"] = "verify"
    except Exception as exc:  # noqa: BLE001
        state["errors"] = state.get("errors", []) + [f"Reasoning failed: {exc}"]
        state["next_action"] = "error"
    return state


def verification_agent(state: AgenticRAGState) -> AgenticRAGState:
    """
    Verification agent (ReAct).

    Thought  → check whether each claim in the draft is supported by context
    Action   → accept, correct, or reject (set GROUNDED + FINAL_ANSWER)
    Observe  → complete, retry retrieval, or stop after max retries

    vs normal RAG: classic RAG has no second agent and no retry loop.
    """
    try:
        docs = state.get("retrieved_docs") or []
        context = format_context(docs)
        draft = state.get("draft_answer") or ""
        prior_thought = state.get("reasoning_thought") or ""

        llm = get_llm(temperature=0.0)
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are the Verification agent in an Agentic RAG system.\n"
                        "Use ReAct: think about grounding, then act by accepting or correcting.\n"
                        "Respond in this exact format:\n"
                        "THOUGHT: <which claims are supported or unsupported>\n"
                        "GROUNDED: yes|no\n"
                        "NOTES: <brief notes>\n"
                        "FINAL_ANSWER: <corrected or confirmed answer>"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{state['query']}\n\n"
                        f"Reasoning agent's THOUGHT:\n{prior_thought}\n\n"
                        f"Retrieved context:\n{context}\n\n"
                        f"Draft answer:\n{draft}"
                    )
                ),
            ]
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = parse_labeled_blocks(
            content, ["THOUGHT", "GROUNDED", "NOTES", "FINAL_ANSWER"]
        )

        thought = parsed.get("THOUGHT") or content
        grounded = "yes" in (parsed.get("GROUNDED") or "").lower()
        notes = parsed.get("NOTES") or ""
        final_answer = parsed.get("FINAL_ANSWER") or draft

        state["verification_thought"] = thought
        state["is_grounded"] = grounded
        state["verification_notes"] = notes
        state["verified_answer"] = final_answer

        if grounded:
            action = "accept_answer"
            observation = "Draft accepted as grounded. Completing workflow."
            state["next_action"] = "complete"
        elif state.get("retry_count", 0) < MAX_RETRIES:
            state["retry_count"] = state.get("retry_count", 0) + 1
            action = "reject_and_retry_retrieval"
            observation = (
                f"Not grounded. Scheduling retrieval retry "
                f"({state['retry_count']}/{MAX_RETRIES})."
            )
            state["errors"] = state.get("errors", []) + [
                f"Verification failed grounding check (retry {state['retry_count']})."
            ]
            state["next_action"] = "retrieve"
        else:
            action = "accept_best_effort"
            observation = "Not grounded after max retries; returning best-effort answer."
            state["errors"] = state.get("errors", []) + [
                "Answer not grounded after max retries; returning best-effort verified answer."
            ]
            state["next_action"] = "complete"

        append_react_step(
            state,
            agent="verification",
            thought=thought,
            action=action,
            observation=observation,
        )
    except Exception as exc:  # noqa: BLE001
        state["errors"] = state.get("errors", []) + [f"Verification failed: {exc}"]
        state["next_action"] = "error"
    return state


def error_handler_agent(state: AgenticRAGState) -> AgenticRAGState:
    """
    Error agent: Act with a safe fallback when Thought/Action fails upstream.

    vs normal RAG: failures often raise or return empty answers with no node.
    """
    errors = state.get("errors") or []
    error_summary = "; ".join(errors) if errors else "Unknown error"
    fallback = (
        state.get("draft_answer")
        or f"Unable to complete Agentic RAG for this query. Errors: {error_summary}"
    )
    state["verified_answer"] = fallback
    state["verification_notes"] = f"Handled by error agent. {error_summary}"
    state["is_grounded"] = False
    append_react_step(
        state,
        agent="error_handler",
        thought="Upstream agent failed; produce a safe fallback instead of crashing.",
        action="emit_fallback_answer",
        observation=error_summary,
    )
    state["next_action"] = "complete"
    return state
