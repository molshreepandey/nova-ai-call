"""Offline ingestion: markdown docs -> heading-aware chunks -> embeddings -> ChromaDB.

Usage:
    python -m rag.ingest            # upsert all docs (idempotent)
    python -m rag.ingest --rebuild  # drop the collection and rebuild from scratch
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from config import CHROMA_DIR, DOCS_DIR, settings, setup_logging
from rag.store import Embedder, get_collection

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 550
OVERLAP_CHARS = 80
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of a document plus its provenance."""

    id: str
    text: str
    source: str
    section: str


def split_sections(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    """Split markdown into ``(title, [(section_heading, body), ...])``.

    The H1 is treated as the document title; every H2+ starts a new section.
    Text before the first H2 becomes an "Overview" section.
    """
    title = ""
    sections: list[tuple[str, str]] = []
    heading = "Overview"
    lines: list[str] = []

    for line in markdown.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            level, text = len(match.group(1)), match.group(2).strip()
            if level == 1 and not title:
                title = text
                continue
            if "\n".join(lines).strip():
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = text, []
        else:
            lines.append(line)
    if "\n".join(lines).strip():
        sections.append((heading, "\n".join(lines).strip()))
    return title, sections


def _blocks(body: str, max_chars: int) -> list[str]:
    """Split a section body into paragraph/list blocks no longer than ``max_chars``."""
    blocks: list[str] = []
    for para in re.split(r"\n\s*\n", body):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            blocks.append(para)
            continue
        # Oversized paragraph or list: fall back to line, then hard, splits.
        for line in para.splitlines():
            line = line.strip()
            while len(line) > max_chars:
                cut = line.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                blocks.append(line[:cut])
                line = line[cut:].strip()
            if line:
                blocks.append(line)
    return blocks


def _tail(text: str, size: int) -> str:
    """Return roughly the last ``size`` characters of ``text``, starting on a word boundary."""
    if len(text) <= size:
        return text
    tail = text[-size:]
    space = tail.find(" ")
    return tail[space + 1 :] if space != -1 else tail


def chunk_text(body: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Greedily pack blocks into chunks of up to ``max_chars``, overlapping by ~``overlap`` chars."""
    chunks: list[str] = []
    current = ""
    for block in _blocks(body, max_chars):
        candidate = f"{current}\n{block}" if current else block
        if len(candidate) <= max_chars:
            current = candidate
            continue
        chunks.append(current)
        carry = _tail(current, overlap)
        current = f"{carry}\n{block}" if len(carry) + len(block) + 1 <= max_chars else block
    if current:
        chunks.append(current)
    return chunks


def chunk_document(path: Path) -> list[Chunk]:
    """Turn one markdown file into chunks with ``source`` and ``section`` metadata.

    Each chunk is prefixed with "title > section" so the embedding carries its context.
    """
    title, sections = split_sections(path.read_text(encoding="utf-8"))
    title = title or path.stem
    chunks: list[Chunk] = []
    for section, body in sections:
        for piece in chunk_text(body):
            chunks.append(
                Chunk(
                    id=f"{path.name}#{len(chunks)}",
                    text=f"[{title} > {section}]\n{piece}",
                    source=path.name,
                    section=section,
                )
            )
    return chunks


def ingest(docs_dir: Path = DOCS_DIR, rebuild: bool = False) -> int:
    """Embed every ``*.md`` file in ``docs_dir`` into Chroma. Returns the collection size.

    Idempotent: each document's existing chunks are deleted before its new chunks
    are added, and chunks of deleted documents are removed, so re-running never
    duplicates and the store always mirrors ``docs_dir``.
    """
    paths = sorted(docs_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No markdown documents found in {docs_dir}")

    collection = get_collection(CHROMA_DIR, settings.collection_name, reset=rebuild)
    embedder = Embedder(settings.embedding_model)

    # Drop chunks of documents that were deleted or renamed since the last run.
    current = {p.name for p in paths}
    stored = {str(m["source"]) for m in collection.get(include=["metadatas"])["metadatas"] or []}
    for stale in sorted(stored - current):
        collection.delete(where={"source": stale})
        logger.info("[INGEST] removed stale document %s", stale)

    for path in paths:
        chunks = chunk_document(path)
        collection.delete(where={"source": path.name})
        collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embedder.embed([c.text for c in chunks]),
            metadatas=[{"source": c.source, "section": c.section} for c in chunks],
        )
        logger.info("[INGEST] %-28s -> %d chunks", path.name, len(chunks))

    total = collection.count()
    logger.info("[INGEST] collection '%s' now holds %d chunks", settings.collection_name, total)
    return total


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Ingest NovaWealth docs into ChromaDB.")
    parser.add_argument("--rebuild", action="store_true", help="drop and rebuild the collection")
    args = parser.parse_args()
    setup_logging()
    ingest(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
