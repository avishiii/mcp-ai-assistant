"""
Bridge between an MCP server and LangChain/LangGraph.

Discovers every tool the MCP server advertises (name, description, JSON schema)
and wraps each one as a LangChain `StructuredTool`, so the LLM can call MCP tools
exactly like native tools. The MCP session stays open for the life of the bridge.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
from langchain_core.tools import StructuredTool


def _result_to_text(result: Any) -> str:
    """Flatten an MCP CallToolResult into a string the LLM can read."""
    structured = getattr(result, "structured_content", None)
    if structured:
        # FastMCP wraps non-object return values as {"result": value}; unwrap them.
        if isinstance(structured, dict) and set(structured) == {"result"}:
            structured = structured["result"]
        if isinstance(structured, str):
            return structured
        return json.dumps(structured, ensure_ascii=False, default=str)
    parts = []
    for block in result.content or []:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else f"[{getattr(block, 'type', 'content')}]")
    return "\n".join(parts) or "(no output)"


class MCPToolBridge:
    """Async context manager that connects to an MCP server and exposes its tools.

    `target` can be:
      * a path to a Python server script  -> launched as a subprocess (stdio transport)
      * an http(s) URL                     -> streamable HTTP transport
      * a FastMCP instance                 -> in-memory transport (great for tests)
    """

    def __init__(self, target: Any):
        if isinstance(target, (str, Path)) and str(target).endswith(".py"):
            # Launch the server as a subprocess with our env (API keys, WORKSPACE_DIR, ...).
            target = PythonStdioTransport(
                script_path=str(target),
                env={**os.environ, "FASTMCP_LOG_LEVEL": "WARNING"},
                python_cmd=sys.executable,
            )
        self.client = Client(target)
        self.tools: list[StructuredTool] = []

    async def __aenter__(self) -> "MCPToolBridge":
        await self.client.__aenter__()
        self.tools = [self._wrap(t) for t in await self.client.list_tools()]
        return self

    async def __aexit__(self, *exc) -> None:
        await self.client.__aexit__(*exc)

    def _wrap(self, mcp_tool: Any) -> StructuredTool:
        name = mcp_tool.name
        schema = getattr(mcp_tool, "input_schema", None) or {"type": "object", "properties": {}}
        client = self.client

        async def _call(**kwargs: Any) -> str:
            # Errors are returned as text so the agent can read them and recover.
            result = await client.call_tool(name, kwargs, raise_on_error=False)
            text = _result_to_text(result)
            return f"ERROR: {text}" if result.is_error else text

        return StructuredTool.from_function(
            coroutine=_call,
            name=name,
            description=mcp_tool.description or name,
            args_schema=schema,
        )
