from langchain_openai import ChatOpenAI

from .config import MODEL_NAME, OPENAI_API_KEY, OPENAI_BASE_URL


def build_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Create an OpenAI or OpenAI-compatible chat model."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and add your key.")

    kwargs = {
        "model": MODEL_NAME or "gpt-4o-mini",
        "api_key": OPENAI_API_KEY,
        "temperature": temperature,
        "timeout": 90,
        "max_retries": 2,
    }
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return ChatOpenAI(**kwargs)
