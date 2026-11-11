from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def get_setting(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    if value is not None and value.strip() == "":
        return None
    return value


MODEL_NAME = get_setting("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = get_setting("OPENAI_API_KEY")
OPENAI_BASE_URL = get_setting("OPENAI_BASE_URL")
TAVILY_API_KEY = get_setting("TAVILY_API_KEY")
MAX_RESEARCH_RESULTS = int(get_setting("MAX_RESEARCH_RESULTS", "5"))
MAX_EDITOR_REVISIONS = int(get_setting("MAX_EDITOR_REVISIONS", "1"))
