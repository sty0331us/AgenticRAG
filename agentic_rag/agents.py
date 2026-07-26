"""
LangGraph agent nodes implementing ReAct (Thought → Action → Observation).

Agents
------
retrieval_agent     Plan search query, execute vector search, record observation
reasoning_agent     Analyze evidence, draft a grounded answer
verification_agent  Validate grounding; accept, correct, or request retry
error_handler_agent Emit a controlled fallback when upstream nodes fail

Compared with classic RAG (retrieve → single generate), drafting and acceptance
are separated, and failures/retries are first-class graph transitions.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_rag.config import get_settings
from agentic_rag.llm import get_llm, invoke_text
from agentic_rag.logging_config import get_logger
from agentic_rag.prompts import (
    REASONING_SYSTEM_PROMPT,
    RETRIEVAL_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
)
from agentic_rag.react import (
    append_error,
    append_react_step,
    format_context,
    parse_labeled_blocks,
)
from agentic_rag.state import AgenticRAGState
from agentic_rag.vectorstore import similarity_search

logger = get_logger(__name__)


def retrieval_agent(state: AgenticRAGState) -> AgenticRAGState:
    """
    Retrieval agent.

    Thought  — formulate an embedding-optimized search query
    Action   — execute similarity search against ChromaDB
    Observe  — persist retrieved evidence and route onward
    """
    cfg = get_settings()
    try:
        llm = get_llm(temperature=cfg.deterministic_temperature, settings=cfg)
        think_text = invoke_text(
            llm,
            [
                SystemMessage(content=RETRIEVAL_SYSTEM_PROMPT),
                HumanMessage(content=f"Question:\n{state['query']}"),
            ],
        )
        parsed = parse_labeled_blocks(think_text, ["THOUGHT", "SEARCH_QUERY"])
        thought = (
            parsed.get("THOUGHT")
            or "Retrieve passages relevant to the submitted question."
        )
        search_query = parsed.get("SEARCH_QUERY") or state["query"]
        state["search_query"] = search_query

        docs, metas = similarity_search(search_query, k=cfg.retrieval_top_k, settings=cfg)
        state["retrieved_docs"] = docs
        state["retrieved_metadatas"] = metas

        observation = (
            f"Retrieved {len(docs)} document(s) for search_query={search_query!r}."
            if docs
            else "Vector store returned zero documents."
        )
        append_react_step(
            state,
            agent="retrieval",
            thought=thought,
            action=f"similarity_search(query={search_query!r}, k={cfg.retrieval_top_k})",
            observation=observation,
        )

        if not docs:
            append_error(
                state,
                "No documents retrieved. Ensure the knowledge base has been ingested.",
            )
            state["next_action"] = "error"
            logger.warning("Retrieval produced no documents")
        else:
            state["next_action"] = "reason"
            logger.info("Retrieval succeeded with %s documents", len(docs))
    except Exception as exc:  # noqa: BLE001 — route to error-handler node
        logger.exception("Retrieval agent failed")
        append_error(state, f"Retrieval failed: {exc}")
        state["next_action"] = "error"
    return state


def reasoning_agent(state: AgenticRAGState) -> AgenticRAGState:
    """
    Reasoning agent.

    Thought  — map evidence to the question and identify gaps
    Action   — produce a draft answer grounded in retrieved context
    Observe  — store draft and advance to verification
    """
    cfg = get_settings()
    try:
        docs = state.get("retrieved_docs") or []
        context = format_context(docs)
        llm = get_llm(temperature=cfg.reasoning_temperature, settings=cfg)
        content = invoke_text(
            llm,
            [
                SystemMessage(content=REASONING_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Question:\n{state['query']}\n\n"
                        f"Search query:\n{state.get('search_query') or state['query']}\n\n"
                        f"Retrieved context:\n{context}"
                    )
                ),
            ],
        )
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
            observation=f"Draft generated ({len(draft)} characters); pending verification.",
        )
        state["next_action"] = "verify"
        logger.info("Reasoning agent produced draft (%s characters)", len(draft))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Reasoning agent failed")
        append_error(state, f"Reasoning failed: {exc}")
        state["next_action"] = "error"
    return state


def verification_agent(state: AgenticRAGState) -> AgenticRAGState:
    """
    Verification agent.

    Thought  — evaluate claim-level grounding against retrieved context
    Action   — accept, correct, or reject the draft
    Observe  — complete the workflow or schedule a retrieval retry
    """
    cfg = get_settings()
    try:
        docs = state.get("retrieved_docs") or []
        context = format_context(docs)
        draft = state.get("draft_answer") or ""
        prior_thought = state.get("reasoning_thought") or ""

        llm = get_llm(temperature=cfg.deterministic_temperature, settings=cfg)
        content = invoke_text(
            llm,
            [
                SystemMessage(content=VERIFICATION_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Question:\n{state['query']}\n\n"
                        f"Reasoning thought:\n{prior_thought}\n\n"
                        f"Retrieved context:\n{context}\n\n"
                        f"Draft answer:\n{draft}"
                    )
                ),
            ],
        )
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

        max_retries = cfg.max_verification_retries
        if grounded:
            action = "accept_answer"
            observation = "Draft accepted as grounded; routing to output guardrail."
            state["next_action"] = "output_guard"
            logger.info("Verification accepted grounded answer")
        elif state.get("retry_count", 0) < max_retries:
            state["retry_count"] = state.get("retry_count", 0) + 1
            action = "reject_and_retry_retrieval"
            observation = (
                f"Grounding check failed; scheduling retrieval retry "
                f"({state['retry_count']}/{max_retries})."
            )
            append_error(
                state,
                f"Verification grounding check failed (retry {state['retry_count']}).",
            )
            state["next_action"] = "retrieve"
            logger.warning("Verification rejected draft; retrying retrieval")
        else:
            action = "accept_best_effort"
            observation = (
                "Grounding check failed after maximum retries; "
                "routing best-effort answer to output guardrail."
            )
            append_error(
                state,
                "Answer not grounded after maximum retries; returning best-effort answer.",
            )
            state["next_action"] = "output_guard"
            logger.warning("Verification exhausted retries; returning best-effort answer")

        append_react_step(
            state,
            agent="verification",
            thought=thought,
            action=action,
            observation=observation,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Verification agent failed")
        append_error(state, f"Verification failed: {exc}")
        state["next_action"] = "error"
    return state


def error_handler_agent(state: AgenticRAGState) -> AgenticRAGState:
    """
    Error-handler agent.

    Emits a controlled fallback response when an upstream node fails, preserving
    auditability instead of propagating an unhandled exception to callers.
    """
    errors = state.get("errors") or []
    error_summary = "; ".join(errors) if errors else "Unknown pipeline error"
    # Preserve messages already set by input/output guardrail nodes.
    fallback = (
        state.get("verified_answer")
        or state.get("draft_answer")
        or f"Unable to complete the request. Details: {error_summary}"
    )
    state["verified_answer"] = fallback
    state["verification_notes"] = f"Resolved by error-handler agent. {error_summary}"
    state["is_grounded"] = False
    append_react_step(
        state,
        agent="error_handler",
        thought="Upstream failure detected; emit a controlled fallback response.",
        action="emit_fallback_answer",
        observation=error_summary,
    )
    state["next_action"] = "complete"
    logger.error("Error-handler agent invoked: %s", error_summary)
    return state
