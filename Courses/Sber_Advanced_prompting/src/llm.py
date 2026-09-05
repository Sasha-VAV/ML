from langchain_openai import ChatOpenAI
from pydantic import SecretStr


def get_model() -> ChatOpenAI:
    """Returns the GigaChat chat model used across the pipeline.

    GigaChat is reached through a locally running `gpt2giga` proxy, which
    exposes an OpenAI-compatible endpoint and translates function-calling and
    structured-output requests to GigaChat's format. Start it before running
    anything: `uv run gpt2giga --port 8090`.
    """
    return ChatOpenAI(
        model="GigaChat-2-Max",
        base_url="http://localhost:8090/v1",
        api_key=SecretStr("not_needed"),
        temperature=0,
    )
