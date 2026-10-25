"""MCP client: spawns the system-tools server over stdio and exposes its tools.

Tool schemas are discovered at runtime with ``list_tools`` and converted to the
OpenAI/Groq function-calling format, so they are never duplicated by hand.
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import AsyncExitStack
from types import TracebackType
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent, Tool

from config import ROOT_DIR

logger = logging.getLogger(__name__)


class MCPToolError(RuntimeError):
    """Raised when an MCP tool reports an error result."""


def _strip_titles(schema: Any) -> Any:
    """Remove pydantic-generated ``title`` keys; they add tokens but no meaning for the LLM."""
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


def _compact_json(text: str) -> str:
    """Re-serialise pretty-printed JSON tool output compactly to save LLM tokens."""
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"))
    except json.JSONDecodeError:
        return text


def mcp_tool_to_openai(tool: Tool) -> dict[str, Any]:
    """Convert an MCP tool description into an OpenAI-style function tool definition."""
    parameters = _strip_titles(tool.inputSchema) or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": (tool.description or "").strip(),
            "parameters": parameters,
        },
    }


class MCPClient:
    """Async context manager owning one stdio MCP server subprocess and its session.

    Example:
        async with MCPClient() as mcp:
            print(mcp.tool_names)
            print(await mcp.call_tool("calculate", {"expression": "2 + 2"}))
    """

    def __init__(self, command: str = sys.executable, args: list[str] | None = None) -> None:
        self._params = StdioServerParameters(
            command=command,
            args=args or ["-m", "mcp_server.server"],
            cwd=str(ROOT_DIR),
        )
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._tools: list[Tool] = []

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def connect(self) -> None:
        """Start the server subprocess, perform the MCP handshake and cache the tool list."""
        read, write = await self._stack.enter_async_context(stdio_client(self._params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        self._tools = (await self._session.list_tools()).tools
        logger.info("[MCP] connected over stdio; tools: %s", ", ".join(self.tool_names))

    async def close(self) -> None:
        """Shut down the session and the server subprocess."""
        await self._stack.aclose()
        self._session = None

    @property
    def tool_names(self) -> list[str]:
        """Names of all tools the server exposes."""
        return [t.name for t in self._tools]

    def openai_tools(self) -> list[dict[str, Any]]:
        """All server tools in OpenAI/Groq function-calling format."""
        return [mcp_tool_to_openai(t) for t in self._tools]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool and return its text output.

        Raises:
            MCPToolError: if the server reports the call as failed.
        """
        if self._session is None:
            raise RuntimeError("MCPClient is not connected")
        logger.info("[MCP] calling %s(%s)", name, arguments)
        result = await self._session.call_tool(name, arguments)
        text = "\n".join(c.text for c in result.content if isinstance(c, TextContent))
        if result.isError:
            raise MCPToolError(text or f"tool {name} failed")
        return _compact_json(text)
