from __future__ import annotations

from typing import TypedDict, Any


class ResearchState(TypedDict, total=False):
    topic: str
    plan: list[str]
    research_queries: list[str]
    research_results: dict[str, str]
    draft: str
    critique: str
    final_report: str
    revision_count: int
    history: list[dict[str, Any]]
    error: str
