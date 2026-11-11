from __future__ import annotations

from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

from .config import MAX_RESEARCH_RESULTS, TAVILY_API_KEY

# These are real external integrations. Tavily requires TAVILY_API_KEY;
# arXiv and Wikipedia are accessible without private API keys.
_arxiv = ArxivQueryRun(top_k_results=MAX_RESEARCH_RESULTS)
_wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=3,
        doc_content_chars_max=5000,
    )
)


@tool

def tavily_search(query: str) -> str:
    """Search the live web for reliable pages about a research query using Tavily."""
    if not TAVILY_API_KEY:
        return "Tavily skipped: TAVILY_API_KEY is not configured."
    try:
        # Initialize lazily so the application can still run without a Tavily key.
        tavily = TavilySearch(
            max_results=MAX_RESEARCH_RESULTS,
            topic="general",
            include_answer=True,
        )
        return str(tavily.invoke({"query": query}))
    except Exception as exc:  # keep the workflow alive if one source is unavailable
        return f"Tavily search error: {type(exc).__name__}: {exc}"


@tool

def arxiv_search(query: str) -> str:
    """Search academic papers on arXiv for a research query."""
    try:
        return str(_arxiv.invoke({"query": query}))
    except Exception as exc:
        return f"arXiv search error: {type(exc).__name__}: {exc}"


@tool

def wikipedia_search(query: str) -> str:
    """Find background information on Wikipedia for a research query."""
    try:
        return str(_wikipedia.invoke({"query": query}))
    except Exception as exc:
        return f"Wikipedia search error: {type(exc).__name__}: {exc}"


RESEARCH_TOOLS = [tavily_search, arxiv_search, wikipedia_search]
