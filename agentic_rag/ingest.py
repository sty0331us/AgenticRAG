"""Knowledge-base ingestion into the vector store."""

from __future__ import annotations

from agentic_rag.config import Settings, get_settings
from agentic_rag.knowledge import KNOWLEDGE_CHUNKS
from agentic_rag.logging_config import get_logger
from agentic_rag.vectorstore import add_documents, collection_count

logger = get_logger(__name__)


def ingest_knowledge(
    force: bool = False,
    settings: Settings | None = None,
) -> int:
    """
    Load seed knowledge chunks into ChromaDB.

    Skips work when the collection is already populated unless ``force`` is True.
    Returns the document count after ingestion.
    """
    cfg = settings or get_settings()
    existing = collection_count(settings=cfg)
    if not force and existing > 0:
        logger.info("Knowledge base already populated (%s documents); skipping ingest", existing)
        return existing

    texts = [chunk["text"] for chunk in KNOWLEDGE_CHUNKS]
    metadatas = [chunk["metadata"] for chunk in KNOWLEDGE_CHUNKS]
    ids = [chunk["id"] for chunk in KNOWLEDGE_CHUNKS]
    add_documents(texts=texts, metadatas=metadatas, ids=ids, settings=cfg)
    total = collection_count(settings=cfg)
    logger.info("Knowledge ingest complete (%s documents)", total)
    return total


if __name__ == "__main__":
    from agentic_rag.logging_config import configure_logging

    configure_logging()
    count = ingest_knowledge(force=True)
    print(f"Ingested {count} documents.")
