"""Semantic search over the ingested knowledge base."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import CHROMA_DIR, settings
from rag.store import Embedder, get_collection

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A search hit with its provenance and cosine similarity (higher is better)."""

    text: str
    source: str
    section: str
    score: float


class Retriever:
    """Embeds a query and returns the closest chunks from ChromaDB."""

    def __init__(self, embedder: Embedder | None = None, collection=None) -> None:  # type: ignore[no-untyped-def]
        self._embedder = embedder or Embedder(settings.embedding_model)
        self._collection = collection or get_collection(CHROMA_DIR, settings.collection_name)

    def warm_up(self) -> None:
        """Load the embedding model now instead of on the first query."""
        self._embedder.embed(["warm up"])

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks ranked by similarity to ``query``."""
        if self._collection.count() == 0:
            raise RuntimeError("Knowledge base is empty. Run `python -m rag.ingest` first.")
        result = self._collection.query(
            query_embeddings=self._embedder.embed([query]),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = [
            RetrievedChunk(
                text=doc,
                source=str(meta["source"]),
                section=str(meta["section"]),
                score=round(1.0 - dist, 3),
            )
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
        logger.info(
            "[RAG] %r -> %s", query, ", ".join(f"{h.source}§{h.section} ({h.score})" for h in hits)
        )
        return hits
