# Agentic Research Workflow — LangChain + LangGraph

A multi-agent research workflow inspired by the supplied assignment. It uses LangChain for model/tool integrations and LangGraph for stateful orchestration.

## Agents

1. **Planner** — creates a 3–4 step research plan.
2. **Research** — collects evidence using Tavily, arXiv, and Wikipedia.
3. **Writer** — writes a source-grounded Markdown report.
4. **Editor** — reviews and improves the report.
5. **Executor / Orchestrator** — the LangGraph state machine that routes execution and maintains shared state/history.

## External tools

- Tavily Search: live web search (`TAVILY_API_KEY` required)
- arXiv: academic paper search (no private key required)
- Wikipedia: background information (no private key required)
- OpenAI-compatible chat API: LLM provider (`OPENAI_API_KEY` required)

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Create your local environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` and `TAVILY_API_KEY`. If using an OpenAI-compatible gateway, set `OPENAI_BASE_URL` and the provider's model name in `MODEL_NAME`. Do not submit `.env` or share API keys.

## Run

From the project root:

```bash
pip install -e .
python -m agentic_research.main "The impact of artificial intelligence on supply chain management"
```

Or, without editable install:

```bash
PYTHONPATH=src python -m agentic_research.main "The impact of artificial intelligence on supply chain management"
```

The workflow writes:

- `reports/research_report.md` — final report
- `reports/research_report_execution.json` — execution history

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | Model API key | Required |
| `OPENAI_BASE_URL` | Optional compatible API endpoint | Official OpenAI endpoint |
| `MODEL_NAME` | Model identifier | `gpt-4o-mini` |
| `TAVILY_API_KEY` | Tavily web search key | Required for Tavily |
| `MAX_RESEARCH_RESULTS` | Max results per source | `5` |
| `MAX_EDITOR_REVISIONS` | Maximum editor passes | `1` |

## Architecture

```text
START
  |
Planner Agent
  |
Research Agent ----> Tavily / arXiv / Wikipedia
  |
Writer Agent
  |
Editor Agent
  |
END ----> Markdown report + JSON execution history
```

LangGraph passes a shared `ResearchState` between nodes. Each agent returns only the state fields it updates. The workflow is intentionally explicit and bounded, so it is easy to explain and debug.

## Notes

- Tool failures are captured as text so a single unavailable source does not immediately stop the entire research workflow.
- The report prompt explicitly prohibits fabricated citations and unsupported facts.
- This is a practical implementation of the assignment's requested roles, not a claim that external API calls have been tested in your environment. Configure valid keys and run it to verify connectivity.
- The assignment handout mentions filling only marked code regions in its original notebook. This is a standalone LangChain/LangGraph implementation, so it changes the implementation approach while preserving the requested agent roles and research tools.
