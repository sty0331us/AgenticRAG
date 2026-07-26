"""
Ingest seed knowledge into ChromaDB.

Difference from normal RAG
--------------------------
Ingest/indexing is the same idea as normal RAG (chunk → embed → store).
Agentic RAG does not change how documents enter the vector DB; it changes what
happens after a query hits that DB (multi-agent graph vs single generate).
"""

from __future__ import annotations

from agentic_rag.knowledge import KNOWLEDGE_CHUNKS
from agentic_rag.vectorstore import add_documents, collection_count


def ingest_knowledge(force: bool = False) -> int:
    """Load knowledge chunks into ChromaDB. Skip if already populated unless force=True."""
    if not force and collection_count() > 0:
        return collection_count()

    texts = [c["text"] for c in KNOWLEDGE_CHUNKS]
    metadatas = [c["metadata"] for c in KNOWLEDGE_CHUNKS]
    ids = [c["id"] for c in KNOWLEDGE_CHUNKS]
    add_documents(texts=texts, metadatas=metadatas, ids=ids)
    return collection_count()


if __name__ == "__main__":
    count = ingest_knowledge(force=True)
    print(f"Ingested {count} documents into ChromaDB.")
