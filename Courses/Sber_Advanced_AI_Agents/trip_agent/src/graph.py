from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from src.llm import get_agent_llm
from src.tools import book_trip, load_skill_content, search_flights, search_hotels, skills_summary

SYSTEM_PROMPT = f"""You are a trip-planning assistant.

You have access to specialized skills, each with detailed instructions for a
specific kind of request. Skills available:

{skills_summary()}

If the user's request matches one of these, call load_skill_content with that
skill's name to load its full instructions, then follow them for the rest of
this reply. If no skill applies (small talk, general questions, anything
outside trip planning), just answer directly and helpfully — don't force a
skill that doesn't fit."""


def build_graph():
    return create_agent(
        get_agent_llm(),
        tools=[load_skill_content, search_flights, search_hotels, book_trip],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
    )
