"""Chunking, retrieval and the search_knowledge_base tool.

A deterministic bag-of-words embedder stands in for MiniLM so tests are fast and
don't download a model; Chroma itself is real (in a temp directory).
"""

import hashlib
import json
import math
import re
from pathlib import Path

import pytest

from config import DOCS_DIR
from rag.ingest import MAX_CHUNK_CHARS, chunk_document, chunk_text, split_sections
from rag.retriever import Retriever
from rag.store import get_collection
from rag.tool import MAX_TOP_K, KnowledgeBaseTool


class HashingEmbedder:
    """Maps each word to a hashed dimension; similar wording -> similar vectors."""

    dims = 256

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dims
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


# --- chunking ---------------------------------------------------------------


def test_split_sections_uses_h1_as_title_and_h2_as_sections() -> None:
    title, sections = split_sections("# Doc\nIntro text.\n\n## Battery\nLasts 14 days.\n\n## Water\n5 ATM.")
    assert title == "Doc"
    assert sections == [("Overview", "Intro text."), ("Battery", "Lasts 14 days."), ("Water", "5 ATM.")]


def test_chunk_text_respects_max_size_and_overlaps() -> None:
    paragraphs = [f"Paragraph {i} " + "word " * 30 for i in range(10)]
    chunks = chunk_text("\n\n".join(paragraphs), max_chars=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    # Each chunk after the first starts with text carried over from the previous one.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.split("\n")[0] in prev


def test_chunk_text_splits_oversized_paragraphs() -> None:
    chunks = chunk_text("x" * 50 + " " + "y" * 2000, max_chars=200)
    assert all(len(c) <= 200 for c in chunks)


def test_every_doc_chunks_with_metadata() -> None:
    paths = sorted(DOCS_DIR.glob("*.md"))
    assert len(paths) == 6
    for path in paths:
        chunks = chunk_document(path)
        assert chunks, path.name
        assert len({c.id for c in chunks}) == len(chunks)  # ids unique -> idempotent upserts
        for c in chunks:
            assert c.source == path.name
            assert c.section
            assert c.text.startswith("[")  # "[title > section]" context prefix
            assert len(c.text) <= MAX_CHUNK_CHARS + 100  # body cap + prefix


# --- retrieval --------------------------------------------------------------


@pytest.fixture
def kb(tmp_path: Path) -> Retriever:
    """A temp Chroma collection holding the real NovaWealth docs."""
    embedder = HashingEmbedder()
    collection = get_collection(tmp_path / "chroma", "test_kb")
    chunks = [c for p in sorted(DOCS_DIR.glob("*.md")) for c in chunk_document(p)]
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embedder.embed([c.text for c in chunks]),
        metadatas=[{"source": c.source, "section": c.section} for c in chunks],
    )
    return Retriever(embedder=embedder, collection=collection)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("query", "expected_source"),
    [
        ("ELSS lock-in Section 80C tax benefit", "mutual_funds.md"),
        ("KYC documents PAN Aadhaar video KYC", "account_opening_kyc.md"),
        ("redemption cut-off time credited business days", "sip_and_redemptions.md"),
        ("PMS fee hurdle rate profit share", "fees_and_charges.md"),
    ],
)
def test_retriever_finds_relevant_doc(kb: Retriever, query: str, expected_source: str) -> None:
    hits = kb.search(query, top_k=3)
    assert hits[0].source == expected_source
    assert hits[0].score >= hits[-1].score


def test_retriever_on_empty_collection_raises(tmp_path: Path) -> None:
    empty = Retriever(embedder=HashingEmbedder(), collection=get_collection(tmp_path, "empty"))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="ingest"):
        empty.search("anything")


def test_rebuild_drops_collection(tmp_path: Path) -> None:
    collection = get_collection(tmp_path, "rebuild_kb")
    collection.add(ids=["a"], documents=["a"], embeddings=[[1.0, 0.0]])
    assert get_collection(tmp_path, "rebuild_kb", reset=True).count() == 0


# --- tool -------------------------------------------------------------------


async def test_tool_returns_json_with_sources(kb: Retriever) -> None:
    tool = KnowledgeBaseTool(kb)
    payload = json.loads(await tool.call({"query": "ELSS lock-in", "top_k": 2}))
    assert len(payload["results"]) == 2
    assert {"source", "section", "score", "text"} <= payload["results"][0].keys()


async def test_tool_clamps_top_k_and_rejects_empty_query(kb: Retriever) -> None:
    tool = KnowledgeBaseTool(kb)
    payload = json.loads(await tool.call({"query": "SIP", "top_k": 999}))
    assert len(payload["results"]) == MAX_TOP_K
    with pytest.raises(ValueError):
        await tool.call({"query": "  "})


def test_tool_definition_is_function_schema() -> None:
    definition = KnowledgeBaseTool(retriever=object()).definition()  # type: ignore[arg-type]
    assert definition["function"]["name"] == "search_knowledge_base"
    assert definition["function"]["parameters"]["required"] == ["query"]
