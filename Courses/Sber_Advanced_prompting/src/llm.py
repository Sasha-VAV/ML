from langchain_openai import ChatOpenAI


def get_model() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-5-mini",
        temperature=0,
    )