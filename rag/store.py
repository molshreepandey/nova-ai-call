"""Shared embedding model and vector-store access for ingestion and retrieval.

Keeping both here guarantees the ingest and query paths always use the same
embedding model and distance metric.
"""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


class Embedder:
    """Lazy wrapper around a local sentence-transformers model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @cached_property
    def _model(self):  # type: ignore[no-untyped-def]
        # Imported lazily: torch import is slow and not needed for every code path.
        from sentence_transformers import SentenceTransformer

        logger.info("[RAG] loading embedding model %s (downloads ~90 MB on first run)", self.model_name)
        return SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return L2-normalised embeddings, one per input text."""
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()


def get_collection(persist_dir: Path, name: str, reset: bool = False) -> Collection:
    """Open (or create) the persistent Chroma collection.

    Args:
        persist_dir: Directory holding the Chroma database.
        name: Collection name.
        reset: Drop the collection first, for a clean rebuild.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )
    if reset and name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
        logger.info("[RAG] dropped collection %s", name)
    # We always supply our own embeddings, so Chroma's default embedder is disabled.
    return client.get_or_create_collection(
        name=name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
    )
