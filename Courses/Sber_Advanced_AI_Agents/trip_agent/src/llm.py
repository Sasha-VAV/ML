import os

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

GIGACHAT_BASE_URL = os.environ.get("GIGACHAT_BASE_URL", "http://localhost:8090/v1")
GIGACHAT_API_KEY = SecretStr("gpt2giga-local")


def get_agent_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="GigaChat-2-Max",
        temperature=0.0,
        base_url=GIGACHAT_BASE_URL,
        api_key=GIGACHAT_API_KEY,
    )


def get_cheap_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="GigaChat-2-Max",
        temperature=0.0,
        base_url=GIGACHAT_BASE_URL,
        api_key=GIGACHAT_API_KEY,
    )
