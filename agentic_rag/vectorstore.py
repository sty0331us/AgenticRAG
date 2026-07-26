"""ChromaDB vector store for Agentic RAG knowledge retrieval."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_COLLECTION = "agentic_rag_knowledge"


def _persist_dir() -> str:
    return os.getenv("CHROMA_PERSIST_DIR", DEFAULT_PERSIST_DIR)


def _collection_name() -> str:
    return os.getenv("CHROMA_COLLECTION", DEFAULT_COLLECTION)


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.ClientAPI:
    """Persistent ChromaDB client (local, no cloud account required)."""
    return chromadb.PersistentClient(
        path=_persist_dir(),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(name: Optional[str] = None):
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name or _collection_name(),
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
) -> None:
    """Upsert documents into the collection (Chroma default embedding)."""
    collection = get_collection()
    if ids is None:
        ids = [f"doc_{i}" for i in range(len(texts))]
    if metadatas is None:
        metadatas = [{"source": "knowledge_base"} for _ in texts]

    collection.upsert(documents=texts, metadatas=metadatas, ids=ids)


def similarity_search(
    query: str,
    k: int = 4,
) -> Tuple[List[str], List[dict]]:
    """Retrieve top-k relevant chunks for a query."""
    collection = get_collection()
    if collection.count() == 0:
        return [], []

    results = collection.query(query_texts=[query], n_results=min(k, collection.count()))
    docs = results.get("documents", [[]])[0] or []
    metas = results.get("metadatas", [[]])[0] or []
    return docs, metas


def collection_count() -> int:
    return get_collection().count()
