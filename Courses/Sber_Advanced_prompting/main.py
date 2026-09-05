import uuid
from pathlib import Path

import asyncio
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from langfuse import get_client

from src.llm import get_model
from src.agent import get_agent
from src.retrieval import load_data, Retrieval
from src.config import Settings
from src.models import Embedder


def main():
    load_dotenv()
    settings = Settings()
    model = get_model()

    data = load_data(Path(__file__).parent / "data" / "SPIEF_txt" / "SPIEF_txt")[:2]
    embedder = Embedder(settings.embedder)
    retrieval = Retrieval(settings.qdrant, embedder)

    retrieval.start(data)

    resp = asyncio.run(retrieval.query(queries=["Перспективы глобальный рынок"], top_k=5))
    agent = get_agent(model)

    langfuse_client = get_client()
    langfuse_handler = CallbackHandler()

    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "callbacks": [langfuse_handler],
    }

    print("Agent started — type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        with langfuse_client.start_as_current_observation(
            name="spief-qa-turn", as_type="span"
        ):
            result = asyncio.run(
                agent.ainvoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config,
                )
            )
        print("=" * 40)
        print(f"\n\nAgent: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
