"""Multi-query retrieval: expand search queries and merge ranked evidence."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agentic_rag.config import Settings
from agentic_rag.logging_config import get_logger
from agentic_rag.react import parse_labeled_blocks
from agentic_rag.retrieval_utils import merge_ranked_hits, parse_alt_queries
from agentic_rag.vectorstore import similarity_search

logger = get_logger(__name__)

__all__ = [
    "generate_alternate_queries",
    "merge_ranked_hits",
    "multi_query_search",
    "parse_alt_queries",
]


def generate_alternate_queries(
    primary_query: str,
    question: str,
    count: int,
    settings: Settings,
) -> Tuple[List[str], str]:
    """
    Ask the LLM for alternate search phrasings.

    Returns ``(queries, thought)``. On failure, returns an empty query list.
    """
    if count <= 0:
        return [], ""

    from langchain_core.messages import HumanMessage, SystemMessage

    from agentic_rag.llm import get_llm, invoke_text
    from agentic_rag.prompts import MULTI_QUERY_SYSTEM_PROMPT

    llm = get_llm(temperature=settings.deterministic_temperature, settings=settings)
    content = invoke_text(
        llm,
        [
            SystemMessage(content=MULTI_QUERY_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Question:\n{question}\n\n"
                    f"Primary search query:\n{primary_query}\n\n"
                    f"Requested alternate count: {count}"
                )
            ),
        ],
    )
    parsed = parse_labeled_blocks(content, ["THOUGHT", "ALT_QUERIES"])
    thought = parsed.get("THOUGHT") or ""
    alts = parse_alt_queries(parsed.get("ALT_QUERIES") or "", count)
    primary_key = primary_query.strip().lower()
    alts = [q for q in alts if q.strip().lower() != primary_key]
    logger.debug("Generated %s alternate search queries", len(alts))
    return alts, thought


def multi_query_search(
    primary_query: str,
    question: str,
    settings: Settings,
) -> Tuple[List[str], List[Dict[str, Any]], List[float], List[str]]:
    """
    Run primary (+ optional alternate) searches and merge top-k unique hits.

    Returns ``(docs, metadatas, scores, queries_used)``.
    """
    queries_used = [primary_query]
    groups: List[Tuple[List[str], List[Dict[str, Any]], List[float]]] = [
        similarity_search(primary_query, k=settings.retrieval_top_k, settings=settings)
    ]

    enabled = settings.multi_query_enabled and settings.multi_query_count > 0
    if enabled:
        try:
            alts, _thought = generate_alternate_queries(
                primary_query,
                question,
                settings.multi_query_count,
                settings,
            )
            for alt in alts:
                queries_used.append(alt)
                groups.append(
                    similarity_search(alt, k=settings.retrieval_top_k, settings=settings)
                )
        except Exception:  # noqa: BLE001 — fall back to primary-only results
            logger.exception("Multi-query expansion failed; using primary query only")

    docs, metas, scores = merge_ranked_hits(groups, settings.retrieval_top_k)
    return docs, metas, scores, queries_used
