"""
ChromaDB persistence and similarity search.

Vector indexing and retrieval mirror classic RAG. Orchestration after retrieval
(specialized agents, routing, verification) is handled by the LangGraph workflow.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from agentic_rag.config import Settings, get_settings
from agentic_rag.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_chroma_client(persist_dir: str) -> chromadb.ClientAPI:
    """Create a persistent ChromaDB client for the given directory."""
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection(
    name: Optional[str] = None,
    settings: Settings | None = None,
) -> Collection:
    """Return (or create) the configured knowledge collection."""
    cfg = settings or get_settings()
    client = get_chroma_client(str(cfg.chroma_persist_dir))
    collection_name = name or cfg.chroma_collection
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(
    texts: Sequence[str],
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ids: Optional[Sequence[str]] = None,
    settings: Settings | None = None,
) -> int:
    """Upsert documents into the collection. Returns number of documents upserted."""
    if not texts:
        return 0

    collection = get_collection(settings=settings)
    doc_ids = list(ids) if ids is not None else [f"doc_{i}" for i in range(len(texts))]
    meta = (
        list(metadatas)
        if metadatas is not None
        else [{"source": "knowledge_base"} for _ in texts]
    )
    if len(doc_ids) != len(texts) or len(meta) != len(texts):
        raise ValueError("texts, ids, and metadatas must have equal length")

    collection.upsert(documents=list(texts), metadatas=meta, ids=doc_ids)
    logger.info("Upserted %s documents into collection '%s'", len(texts), collection.name)
    return len(texts)


def _distance_to_score(distance: Optional[float]) -> float:
    """Convert Chroma cosine distance to a similarity score in roughly [0, 1]."""
    if distance is None:
        return 0.0
    return max(0.0, 1.0 - float(distance))


def similarity_search(
    query: str,
    k: Optional[int] = None,
    settings: Settings | None = None,
) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
    """
    Retrieve top-k documents by embedding similarity.

    Returns:
        (documents, metadatas, similarity_scores)
        Scores are derived from cosine distance as ``1 - distance``.
    """
    cfg = settings or get_settings()
    top_k = k if k is not None else cfg.retrieval_top_k
    collection = get_collection(settings=cfg)
    total = collection.count()
    if total == 0:
        logger.warning("Similarity search requested against an empty collection")
        return [], [], []

    n_results = min(top_k, total)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    docs = (results.get("documents") or [[]])[0] or []
    metas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []
    scores = [_distance_to_score(d) for d in distances]
    # Keep lists aligned if Chroma omits a field for any hit.
    while len(metas) < len(docs):
        metas.append({})
    while len(scores) < len(docs):
        scores.append(0.0)
    logger.debug("Retrieved %s/%s documents for query", len(docs), n_results)
    return docs, metas, scores


def collection_count(settings: Settings | None = None) -> int:
    """Return the number of documents currently indexed."""
    return get_collection(settings=settings).count()
