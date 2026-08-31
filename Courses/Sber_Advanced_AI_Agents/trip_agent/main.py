import asyncio
import uuid

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler

load_dotenv()


async def main():
    from src.graph import build_graph
    graph = build_graph()
    langfuse_handler = CallbackHandler()
    langfuse_client = get_client()
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "callbacks": [langfuse_handler],
    }

    print("Trip agent — type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        with langfuse_client.start_as_current_observation(name="trip_agent_turn", as_type="span"):
            result = await graph.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
        print("=" * 40)
        print(f"\n\nAgent: {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
