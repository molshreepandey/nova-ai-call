"""The ``search_knowledge_base`` tool: RAG exposed to the LLM as a callable tool.

The LLM decides when to call this. It runs inside the agent process, next to
the orchestrator, while system tools live in the separate MCP server.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from config import settings
from rag.retriever import Retriever

MAX_TOP_K = 8


class KnowledgeBaseTool:
    """Local tool that searches NovaWealth's product and policy documents."""

    name = "search_knowledge_base"
    description = (
        "Search NovaWealth's internal knowledge base (mutual funds and their fees, exit "
        "loads and lock-ins; SIPs, redemptions and switches; account opening and KYC; "
        "advisory and PMS fees; risk, disclaimers and grievances; company contact details). "
        "Use it for any NovaWealth or investing question. Write the query in English. Returns text chunks "
        "with their source document and section."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A focused natural-language search query, e.g. 'ELSS lock-in period'.",
            },
            "top_k": {
                "type": "integer",
                "description": f"Number of chunks to return (1-{MAX_TOP_K}). Default {settings.rag_top_k}.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, retriever: Retriever | None = None) -> None:
        self._retriever = retriever or Retriever()

    def definition(self) -> dict[str, Any]:
        """Tool schema in OpenAI/Groq function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def call(self, arguments: dict[str, Any]) -> str:
        """Run the search and return a JSON string for the LLM.

        Retrieval is CPU-bound (embedding), so it runs in a worker thread to keep
        the event loop free.
        """
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValueError("'query' must be a non-empty string")
        top_k = max(1, min(int(arguments.get("top_k") or settings.rag_top_k), MAX_TOP_K))
        hits = await asyncio.to_thread(self._retriever.search, query, top_k)
        return json.dumps(
            {
                "results": [
                    {"source": h.source, "section": h.section, "score": h.score, "text": h.text}
                    for h in hits
                ]
            },
            ensure_ascii=False,
        )
