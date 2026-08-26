import asyncio
import os
import uuid

from dotenv import load_dotenv

from langfuse.langchain import CallbackHandler


load_dotenv()


async def main():
    from src.graph import build_graph
    graph = build_graph()
    langfuse_handler = CallbackHandler()
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "callbacks": [langfuse_handler],
    }

    print("Trip agent — type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        print(f"Agent: {result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
