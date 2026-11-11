from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from .agents import planner_agent, research_agent, writer_agent, editor_agent
from .state import ResearchState


def build_workflow():
    """Build the stateful multi-agent workflow using LangGraph."""
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_agent)
    graph.add_node("research", research_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("editor", editor_agent)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "writer")
    graph.add_edge("writer", "editor")
    graph.add_edge("editor", END)

    return graph.compile()
