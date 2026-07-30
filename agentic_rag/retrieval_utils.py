"""Pure helpers for multi-query retrieval (no vector-store / LLM imports)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple


def parse_alt_queries(raw: str, limit: int) -> List[str]:
    """Parse pipe- or newline-separated alternate queries, capped at ``limit``."""
    if not raw or limit <= 0:
        return []
    parts: List[str] = []
    for chunk in raw.replace("\n", "|").split("|"):
        cleaned = chunk.strip().strip("-").strip()
        if cleaned:
            parts.append(cleaned)
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
