from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from .llm import build_llm
from .state import ResearchState
from .tools import tavily_search, arxiv_search, wikipedia_search
from .config import MAX_EDITOR_REVISIONS


def _record(state: ResearchState, agent: str, detail: str) -> list[dict]:
    history = list(state.get("history", []))
    history.append({
        "agent": agent,
        "detail": detail,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return history


def _content(response) -> str:
    content = response.content
    return content if isinstance(content, str) else str(content)


def _parse_plan(text: str) -> list[str]:
    """Accept JSON arrays or numbered/bulleted plans from different providers."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = data.get("plan", data.get("steps", []))
        if isinstance(data, list):
            items = [str(item).strip() for item in data if str(item).strip()]
            if items:
                return items[:4]
    except (json.JSONDecodeError, TypeError):
        pass

    lines = []
    for line in cleaned.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line:
            lines.append(line)
    return lines[:4] or [
        "Collect recent and foundational research about the topic",
        "Compare the main methods, findings, and limitations",
        "Synthesize the evidence into a structured report",
    ]


def planner_agent(state: ResearchState) -> dict:
    """Create a concise research plan (the Planner Agent in the assignment)."""
    llm = build_llm(temperature=0.2)
    topic = state["topic"]
    prompt = f"""Create a research plan for this topic: {topic}
Return ONLY a JSON array containing 3 or 4 short, actionable research steps.
The steps should cover: background/definitions, academic evidence, current web evidence,
and synthesis. Do not write the report yet."""
    response = llm.invoke([HumanMessage(content=prompt)])
    plan = _parse_plan(_content(response))
    return {"plan": plan, "history": _record(state, "planner", f"Created {len(plan)} research steps.")}


def research_agent(state: ResearchState) -> dict:
    """Collect external evidence from Tavily, arXiv, and Wikipedia."""
    topic = state["topic"]
    plan = state.get("plan", [])
    # One focused query per source avoids unnecessary API usage and keeps results readable.
    queries = [
        f"{topic}: recent web research, reports, and authoritative sources",
        f"{topic}: academic papers, methods, and empirical findings",
        f"{topic}: definition, background, and key concepts",
    ]
    results = {
        "tavily": tavily_search.invoke({"query": queries[0]}),
        "arxiv": arxiv_search.invoke({"query": queries[1]}),
        "wikipedia": wikipedia_search.invoke({"query": queries[2]}),
    }
    history = _record(state, "research", "Queried Tavily, arXiv, and Wikipedia.")
    return {"research_queries": queries, "research_results": results, "history": history}


def writer_agent(state: ResearchState) -> dict:
    """Write a source-grounded first draft from the collected evidence."""
    llm = build_llm(temperature=0.5)
    evidence = state.get("research_results", {})
    plan = state.get("plan", [])
    prompt = f"""Write a concise, well-structured research report in the same language as the topic.
Topic: {state['topic']}
Research plan: {json.dumps(plan, ensure_ascii=False)}
Evidence gathered from external tools:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Requirements:
- Follow the plan and use clear headings.
- Distinguish sourced facts from interpretation.
- Do not invent citations, URLs, paper titles, dates, or statistics.
- Mention when a source returned no useful result or was unavailable.
- Include a short conclusion and a Sources section listing only URLs/titles actually present in the evidence.
"""
    response = llm.invoke([
        SystemMessage(content="You are a careful academic and technical research writer. Never fabricate sources."),
        HumanMessage(content=prompt),
    ])
    draft = _content(response)
    return {"draft": draft, "history": _record(state, "writer", "Produced the first report draft.")}


def editor_agent(state: ResearchState) -> dict:
    """Critique and improve the draft; keep review bounded to avoid endless loops."""
    llm = build_llm(temperature=0.1)
    # On a second pass, review the previous edited version instead of the original draft.
    current = state.get('final_report') or state.get('draft', '')
    prompt = f"""Review and edit the following research report.
Topic: {state['topic']}
Evidence: {json.dumps(state.get('research_results', {}), ensure_ascii=False)}
Draft:
{current}

Check factual support, structure, clarity, repetition, and whether sources are represented honestly.
Do not add unsupported facts or invented references. Return the complete improved report only.
"""
    response = llm.invoke([
        SystemMessage(content="You are a strict academic editor. Preserve supported claims and remove unsupported ones."),
        HumanMessage(content=prompt),
    ])
    edited = _content(response)
    count = int(state.get("revision_count", 0)) + 1
    return {
        "final_report": edited,
        "revision_count": count,
        "critique": "The editor reviewed factual support, clarity, structure, and references.",
        "history": _record(state, "editor", f"Completed review pass {count}.")
    }


def should_review_again(state: ResearchState) -> str:
    """Conditional route used by LangGraph to cap editor iterations."""
    # One review is normally enough. The setting allows a second pass if desired.
    if int(state.get("revision_count", 0)) < MAX_EDITOR_REVISIONS:
        return "editor"
    return "finish"
