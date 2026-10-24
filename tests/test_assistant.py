"""Offline tests: no OpenAI key or network needed."""
import importlib
import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("NOTES_DB", str(tmp_path / "notes.db"))
    import server.mcp_server as mod
    return importlib.reload(mod).mcp


@pytest.fixture
async def bridge(server):
    from agent.mcp_bridge import MCPToolBridge
    async with MCPToolBridge(server) as b:
        yield b


def tool(bridge, name):
    return next(t for t in bridge.tools if t.name == name)


async def test_tools_are_discovered(bridge):
    names = {t.name for t in bridge.tools}
    assert {"read_file", "write_file", "list_files", "add_note", "search_notes",
            "get_weather", "wikipedia_summary", "calculate"} <= names


async def test_file_roundtrip(bridge):
    await tool(bridge, "write_file").ainvoke({"path": "a/hello.txt", "content": "hi there"})
    assert "a/hello.txt" in await tool(bridge, "list_files").ainvoke({})
    assert await tool(bridge, "read_file").ainvoke({"path": "a/hello.txt"}) == "hi there"


async def test_write_refuses_overwrite(bridge):
    w = tool(bridge, "write_file")
    await w.ainvoke({"path": "x.txt", "content": "1"})
    assert (await w.ainvoke({"path": "x.txt", "content": "2"})).startswith("ERROR")


async def test_path_traversal_blocked(bridge):
    out = await tool(bridge, "read_file").ainvoke({"path": "../../etc/passwd"})
    assert out.startswith("ERROR") and "outside the workspace" in out


async def test_notes(bridge):
    await tool(bridge, "add_note").ainvoke({"title": "Groceries", "body": "milk, eggs"})
    found = json.loads(await tool(bridge, "search_notes").ainvoke({"query": "milk"}))
    assert found[0]["title"] == "Groceries"


@pytest.mark.parametrize("expr,expected", [("2+3*4", 14), ("(10-4)/3", 2), ("2**10", 1024)])
async def test_calculate(bridge, expr, expected):
    assert float(await tool(bridge, "calculate").ainvoke({"expression": expr})) == expected


async def test_calculate_rejects_code(bridge):
    out = await tool(bridge, "calculate").ainvoke({"expression": "__import__('os').system('ls')"})
    assert out.startswith("ERROR")


class ScriptedLLM(GenericFakeChatModel):
    """Fake chat model that replays scripted messages and accepts bind_tools()."""
    def bind_tools(self, tools, **kwargs):
        return self


async def test_agent_loop_calls_mcp_tool(bridge):
    from agent.graph import build_graph
    llm = ScriptedLLM(messages=iter([
        AIMessage("", tool_calls=[{"name": "calculate", "args": {"expression": "6*7"}, "id": "c1"}]),
        AIMessage("The answer is 42."),
    ]))
    graph = build_graph(llm, bridge.tools)
    result = await graph.ainvoke(
        {"messages": [HumanMessage("What is 6*7?")]},
        {"configurable": {"thread_id": "t1"}},
    )
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert float(tool_msgs[0].content) == 42
    assert result["messages"][-1].content == "The answer is 42."
