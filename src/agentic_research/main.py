from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .graph import build_workflow


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate a research report with LangChain + LangGraph.")
    parser.add_argument("topic", nargs="*", help="Research topic")
    parser.add_argument("--output", default="reports/research_report.md", help="Output Markdown file")
    args = parser.parse_args()

    topic = " ".join(args.topic).strip()
    if not topic:
        topic = input("Enter a research topic: ").strip()
    if not topic:
        raise SystemExit("A research topic is required.")

    workflow = build_workflow()
    result = workflow.invoke({
        "topic": topic,
        "history": [],
        "revision_count": 0,
    })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.get("final_report", "No report was generated."), encoding="utf-8")

    history_path = output_path.with_name(output_path.stem + "_execution.json")
    history_path.write_text(
        json.dumps(result.get("history", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== RESEARCH PLAN ===")
    for index, step in enumerate(result.get("plan", []), start=1):
        print(f"{index}. {step}")
    print(f"\nReport saved to: {output_path.resolve()}")
    print(f"Execution history saved to: {history_path.resolve()}")


if __name__ == "__main__":
    main()
