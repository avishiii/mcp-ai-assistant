"""
LangGraph agent: a ReAct-style loop built explicitly as a state graph.

        ┌──────────┐   tool calls?   ┌─────────┐
START ─►│  agent   │ ───── yes ────► │  tools  │
        │  (LLM)   │ ◄────────────── │  (MCP)  │
        └────┬─────┘                 └─────────┘
             │ no
             ▼
            END

Conversation memory is kept per `thread_id` with a checkpointer.
"""
from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

SYSTEM_PROMPT = """You are a helpful AI assistant connected to external tools through the \
Model Context Protocol (MCP).

Guidelines:
- Use tools whenever they give a more accurate or up-to-date answer (weather, files, notes, math, facts).
- You may call several tools in sequence; plan the steps, then act.
- File paths are relative to the user's workspace folder. List files first if unsure.
- If a tool returns an ERROR, explain it briefly and try a sensible alternative.
- Never invent tool results. Keep final answers concise and clear.

Current UTC time: {now}"""


def build_graph(llm: BaseChatModel, tools: list[BaseTool], checkpointer=None):
    """Compile the agent graph for the given chat model and tools."""
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: MessagesState) -> dict:
        system = SystemMessage(
            SYSTEM_PROMPT.format(now=datetime.now(timezone.utc).isoformat(timespec="minutes"))
        )
        response = await llm_with_tools.ainvoke([system, *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
