"""
Interactive CLI for the MCP-powered assistant.

    python main.py                                  # launches the MCP server over stdio
    python main.py --server http://127.0.0.1:8000/mcp   # connect to a running HTTP server
    python main.py --ask "What's the weather in Pune?"  # one-shot question
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent.graph import build_graph
from agent.mcp_bridge import MCPToolBridge

DEFAULT_SERVER = str(Path(__file__).parent / "server" / "mcp_server.py")
console = Console()


async def run_turn(graph, text: str, thread_id: str, verbose: bool) -> None:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    final = None
    async for update in graph.astream(
        {"messages": [HumanMessage(text)]}, config, stream_mode="updates"
    ):
        for node, payload in update.items():
            for msg in payload.get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for call in msg.tool_calls:
                        console.print(f"[cyan]→ tool[/cyan] [bold]{call['name']}[/bold] {call['args']}")
                elif isinstance(msg, ToolMessage) and verbose:
                    preview = str(msg.content)
                    console.print(f"[dim]← {preview[:300]}{'…' if len(preview) > 300 else ''}[/dim]")
                elif isinstance(msg, AIMessage):
                    final = msg.content
    if final:
        console.print(Panel(Markdown(final), title="Assistant", border_style="green"))


async def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="MCP-powered AI assistant")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Server script path or HTTP URL")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--ask", help="Ask a single question and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show tool outputs")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.[/red]")
        sys.exit(1)

    llm = ChatOpenAI(model=args.model, temperature=0)

    async with MCPToolBridge(args.server) as bridge:
        graph = build_graph(llm, bridge.tools)
        thread_id = str(uuid.uuid4())
        console.print(
            Panel(
                f"Connected to MCP server with [bold]{len(bridge.tools)}[/bold] tools: "
                + ", ".join(t.name for t in bridge.tools)
                + f"\nModel: {args.model}   •   type 'exit' to quit, 'reset' for a new conversation",
                title="MCP AI Assistant",
                border_style="blue",
            )
        )

        if args.ask:
            await run_turn(graph, args.ask, thread_id, args.verbose)
            return

        while True:
            try:
                text = console.input("[bold yellow]You:[/bold yellow] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text.lower() in {"exit", "quit"}:
                break
            if text.lower() == "reset":
                thread_id = str(uuid.uuid4())
                console.print("[dim]Started a new conversation.[/dim]")
                continue
            try:
                await run_turn(graph, text, thread_id, args.verbose)
            except Exception as exc:  # keep the REPL alive on API/network errors
                console.print(f"[red]Error:[/red] {exc}")


if __name__ == "__main__":
    asyncio.run(main())
