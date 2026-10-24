"""
MCP server built with FastMCP.

Exposes three kinds of MCP primitives:
  * Tools     - file operations, notes (SQLite), weather API, Wikipedia API, calculator
  * Resources - notes://all  (read-only view of every saved note)
  * Prompts   - summarize_file (reusable prompt template)

Run standalone:
    python server/mcp_server.py               # stdio transport (default)
    python server/mcp_server.py --http 8000   # streamable HTTP transport
"""
from __future__ import annotations

import ast
import operator
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.getenv("WORKSPACE_DIR", ROOT / "workspace")).resolve()
DB_PATH = Path(os.getenv("NOTES_DB", ROOT / "notes.db")).resolve()
HTTP_TIMEOUT = 10.0
MAX_READ_CHARS = 20_000

WORKSPACE.mkdir(parents=True, exist_ok=True)

mcp = FastMCP(
    name="assistant-tools",
    instructions=(
        "Tools for a personal AI assistant: sandboxed file access, persistent notes, "
        "live weather, Wikipedia lookups and safe math."
    ),
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _safe_path(relative: str) -> Path:
    """Resolve a path inside the workspace and block path-traversal attempts."""
    target = (WORKSPACE / relative).resolve()
    if target != WORKSPACE and WORKSPACE not in target.parents:
        raise ValueError(f"Access denied: '{relative}' is outside the workspace.")
    return target


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS notes ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, "
        "body TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    return conn


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large.")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Only numbers and + - * / // % ** ( ) are allowed.")


def _search_notes(query: str = "", limit: int = 10) -> list[dict]:
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, title, body, created_at FROM notes "
            "WHERE title LIKE ? OR body LIKE ? ORDER BY id DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", max(1, min(limit, 50))),
        ).fetchall()
    return [dict(zip(("id", "title", "body", "created_at"), r)) for r in rows]


# --------------------------------------------------------------------------- #
# File tools
# --------------------------------------------------------------------------- #
@mcp.tool
def list_files(subdir: str = ".") -> list[str]:
    """List files inside the assistant's workspace folder (or a subfolder of it)."""
    base = _safe_path(subdir)
    if not base.is_dir():
        raise ValueError(f"'{subdir}' is not a directory.")
    return sorted(str(p.relative_to(WORKSPACE)) for p in base.rglob("*") if p.is_file())


@mcp.tool
def read_file(path: str) -> str:
    """Read a text file from the workspace. `path` is relative to the workspace."""
    target = _safe_path(path)
    if not target.is_file():
        raise ValueError(f"File not found: '{path}'.")
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + "\n...[truncated]"
    return text


@mcp.tool
def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Create a text file in the workspace. Set overwrite=True to replace an existing file."""
    target = _safe_path(path)
    if target.exists() and not overwrite:
        raise ValueError(f"'{path}' already exists. Pass overwrite=True to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to '{path}'."


# --------------------------------------------------------------------------- #
# Notes tools (persistent, SQLite)
# --------------------------------------------------------------------------- #
@mcp.tool
def add_note(title: str, body: str) -> str:
    """Save a note that persists between sessions."""
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO notes (title, body, created_at) VALUES (?, ?, ?)",
            (title, body, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
    return f"Saved note #{cur.lastrowid}: {title}"


@mcp.tool
def search_notes(query: str = "", limit: int = 10) -> list[dict]:
    """Search saved notes by keyword in title or body. Empty query returns the latest notes."""
    return _search_notes(query, limit)


@mcp.tool
def delete_note(note_id: int) -> str:
    """Delete a note by its id."""
    with _db() as conn:
        cur = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    if cur.rowcount == 0:
        raise ValueError(f"No note with id {note_id}.")
    return f"Deleted note #{note_id}."


# --------------------------------------------------------------------------- #
# External API tools
# --------------------------------------------------------------------------- #
@mcp.tool
async def get_weather(city: str) -> dict:
    """Get the current weather for a city (Open-Meteo API, no key needed)."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        geo = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
        )
        geo.raise_for_status()
        results = geo.json().get("results")
        if not results:
            raise ValueError(f"Could not find a city named '{city}'.")
        place = results[0]
        wx = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            },
        )
        wx.raise_for_status()
    current = wx.json()["current"]
    return {
        "location": f"{place['name']}, {place.get('country', '')}".strip(", "),
        "temperature_c": current["temperature_2m"],
        "humidity_pct": current["relative_humidity_2m"],
        "wind_kmh": current["wind_speed_10m"],
        "weather_code": current["weather_code"],
        "observed_at": current["time"],
    }


@mcp.tool
async def wikipedia_summary(topic: str) -> dict:
    """Get a short summary of a topic from Wikipedia."""
    headers = {"User-Agent": "mcp-ai-assistant/1.0 (educational project)"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=headers) as client:
        search = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": topic, "limit": 1, "format": "json"},
        )
        search.raise_for_status()
        titles = search.json()[1]
        if not titles:
            raise ValueError(f"No Wikipedia article found for '{topic}'.")
        page = await client.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + titles[0].replace(" ", "_")
        )
        page.raise_for_status()
    data = page.json()
    return {
        "title": data.get("title"),
        "summary": data.get("extract"),
        "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
    }


# --------------------------------------------------------------------------- #
# Utility tools
# --------------------------------------------------------------------------- #
@mcp.tool
def calculate(expression: str) -> float:
    """Safely evaluate an arithmetic expression, e.g. '(12.5 * 4) / 3 + 2**8'."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {expression}") from exc
    return _eval(tree)


@mcp.tool
def current_time() -> str:
    """Return the current UTC date and time in ISO format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Resources & prompts
# --------------------------------------------------------------------------- #
@mcp.resource("notes://all")
def all_notes() -> str:
    """Every saved note as plain text."""
    notes = _search_notes("", limit=50)
    if not notes:
        return "No notes yet."
    return "\n\n".join(f"#{n['id']} {n['title']} ({n['created_at']})\n{n['body']}" for n in notes)


@mcp.prompt
def summarize_file(path: str) -> str:
    """Prompt template asking the assistant to summarize a workspace file."""
    return (
        f"Read the file '{path}' with the read_file tool and give a concise summary "
        "with the 3 most important points as bullets."
    )


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--http":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        mcp.run(transport="http", host="127.0.0.1", port=port)
    else:
        mcp.run(show_banner=False)  # stdio: keep the client's terminal clean
