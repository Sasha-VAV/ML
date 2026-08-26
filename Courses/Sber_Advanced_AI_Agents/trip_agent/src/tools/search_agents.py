from langchain.agents import create_agent
from langchain_core.tools import tool

from src.llm import get_agent_llm
from src.tools.providers import find_hotel, find_tickets

_FLIGHTS_SYSTEM_PROMPT = """You are a narrow flight-search assistant. You only search flights
— you don't discuss hotels, destination choice, or booking. Use the find_tickets tool with
whatever trip details you're given. If a critical detail is missing (destination or dates),
ask one short clarifying question instead of guessing. Once you have results, report them
concisely."""

_HOTELS_SYSTEM_PROMPT = """You are a narrow hotel-search assistant. You only search hotels —
you don't discuss flights, destination choice, or booking. Use the find_hotel tool with
whatever trip details you're given. If a critical detail is missing (destination or dates),
ask one short clarifying question instead of guessing. Once you have results, report them
concisely."""

_flights_agent = create_agent(get_agent_llm(), tools=[find_tickets], system_prompt=_FLIGHTS_SYSTEM_PROMPT)
_hotels_agent = create_agent(get_agent_llm(), tools=[find_hotel], system_prompt=_HOTELS_SYSTEM_PROMPT)


@tool
async def search_flights(request: str) -> str:
    """Delegate to a specialized flight-search agent. Pass the trip details you have so far
    (destination, dates, party size, budget) as `request`. It may return concrete flight
    options, or a clarifying question if something critical is missing."""
    result = await _flights_agent.ainvoke(
        {"messages": [{"role": "user", "content": request}]},
        config={"recursion_limit": 6},
    )
    return result["messages"][-1].content


@tool
async def search_hotels(request: str) -> str:
    """Delegate to a specialized hotel-search agent. Pass the trip details you have so far
    (destination, dates, party size, budget) as `request`. It may return concrete hotel
    options, or a clarifying question if something critical is missing."""
    result = await _hotels_agent.ainvoke(
        {"messages": [{"role": "user", "content": request}]},
        config={"recursion_limit": 6},
    )
    return result["messages"][-1].content
