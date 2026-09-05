from langchain_openai import ChatOpenAI
from pydantic import SecretStr


def get_model() -> ChatOpenAI:
    return ChatOpenAI(
        model="GigaChat-2-Max",
        base_url="http://localhost:8090/v1",
        api_key=SecretStr("not_needed"),
        temperature=0,
    )
