"""Multi-query retrieval helpers: expand, search, and merge ranked evidence."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_rag.config import Settings
from agentic_rag.llm import get_llm, invoke_text
from agentic_rag.logging_config import get_logger
from agentic_rag.prompts import MULTI_QUERY_SYSTEM_PROMPT
from agentic_rag.react import parse_labeled_blocks
from agentic_rag.vectorstore import similarity_search

logger = get_logger(__name__)


def parse_alt_queries(raw: str, limit: int) -> List[str]:
    """Parse pipe- or newline-separated alternate queries, capped at ``limit``."""
    if not raw or limit <= 0:
        return []
    parts: List[str] = []
    for chunk in raw.replace("\n", "|").split("|"):
        cleaned = chunk.strip().strip("-").strip()
        if cleaned:
            parts.append(cleaned)
    # Deduplicate while preserving order.
    seen = set()
    unique: List[str] = []
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(part)
        if len(unique) >= limit:
            break
    return unique


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
    # Drop alternates that duplicate the primary query.
    primary_key = primary_query.strip().lower()
    alts = [q for q in alts if q.strip().lower() != primary_key]
    logger.debug("Generated %s alternate search queries", len(alts))
    return alts, thought


def merge_ranked_hits(
    hit_groups: Sequence[Tuple[List[str], List[Dict[str, Any]], List[float]]],
    top_k: int,
) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
    """
    Merge retrieval results across queries, keeping the highest score per document.

    Documents are keyed by exact text content.
    """
    best: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    for docs, metas, scores in hit_groups:
        for i, doc in enumerate(docs):
            score = scores[i] if i < len(scores) else 0.0
            meta = dict(metas[i]) if i < len(metas) else {}
            previous = best.get(doc)
            if previous is None or score > previous[0]:
                best[doc] = (score, meta)

    ranked = sorted(best.items(), key=lambda item: item[1][0], reverse=True)[:top_k]
    merged_docs = [doc for doc, _ in ranked]
    merged_scores = [score for _, (score, _) in ranked]
    merged_metas = [meta for _, (_, meta) in ranked]
    return merged_docs, merged_metas, merged_scores


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
