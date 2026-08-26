from langchain_openai import ChatOpenAI

def get_agent_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-5.4",
        temperature=0.0,
    )
