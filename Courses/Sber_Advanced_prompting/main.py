import uuid
from pathlib import Path

import asyncio
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from langfuse import get_client

from src.llm import get_model
from src.agent import get_agent
from src.context import AgentContext
from src.retrieval import load_data, Retrieval
from src.config import Settings
from src.models import Embedder


async def main():
    load_dotenv()
    settings = Settings()

    data = load_data(Path(__file__).parent / "data" / "SPIEF_txt" / "SPIEF_txt")
    if settings.max_documents is not None:
        data = data[: settings.max_documents]

    embedder = Embedder(settings.embedder)
    retrieval = Retrieval(settings.qdrant, embedder)
    await retrieval.start(data)

    agent = get_agent(get_model())
    context = AgentContext(retrieval=retrieval)

    langfuse_client = get_client()
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "callbacks": [CallbackHandler()],
    }

    print("Agent started — type 'exit' to quit.")
    while True:
        user_input = await asyncio.to_thread(input, "You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        with langfuse_client.start_as_current_observation(
            name="spief-qa-turn", as_type="span"
        ):
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                context=context,
            )
        print("=" * 40)
        print(f"\n\nAgent: {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
