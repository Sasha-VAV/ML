from typing import Annotated

from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState

from src.llm import get_cheap_llm
from src.tools.providers import find_hotel, find_tickets

_FLIGHTS_SYSTEM_PROMPT = """You are a narrow flight-search assistant. You only search flights
— you don't discuss hotels, destination choice, or booking. Use the find_tickets tool with
whatever trip details you're given. If a critical detail is missing (destination or dates),
ask one short clarifying question instead of guessing. Once you have results, report them
concisely. If a search comes back saying it's temporarily unavailable, you may try it again
once yourself if you think the issue could be transient — otherwise tell the traveler it's
not available right now."""

_HOTELS_SYSTEM_PROMPT = """You are a narrow hotel-search assistant. You only search hotels —
you don't discuss flights, destination choice, or booking. Use the find_hotel tool with
whatever trip details you're given. If a critical detail is missing (destination or dates),
ask one short clarifying question instead of guessing. Once you have results, report them
concisely. If a search comes back saying it's temporarily unavailable, you may try it again
once yourself if you think the issue could be transient — otherwise tell the traveler it's
not available right now."""


def _retry_message(exc: BaseException) -> str:
    return f"Search failed after retries ({exc}). Temporarily unavailable."


def _child_had_error(messages: list) -> bool:
    return any(isinstance(m, ToolMessage) and m.status == "error" for m in messages)


_flights_agent = create_agent(
    get_cheap_llm(),
    tools=[find_tickets],
    system_prompt=_FLIGHTS_SYSTEM_PROMPT,
    middleware=[ToolRetryMiddleware(max_retries=2, tools=["find_tickets"], on_failure=_retry_message)],
)
_hotels_agent = create_agent(
    get_cheap_llm(),
    tools=[find_hotel],
    system_prompt=_HOTELS_SYSTEM_PROMPT,
    middleware=[ToolRetryMiddleware(max_retries=2, tools=["find_hotel"], on_failure=_retry_message)],
)


@tool
async def search_flights(
    request: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    destination: Annotated[str | None, InjectedState("destination")] = None,
) -> ToolMessage:
    """Delegate to a specialized flight-search agent. Pass the trip details you have so far
    (destination, dates, party size, budget) as `request`. It may return concrete flight
    options, or a clarifying question if something critical is missing."""
    if not destination:
        return ToolMessage(
            content="Rejected: no destination confirmed yet — call confirm_destination first.",
            status="error",
            tool_call_id=tool_call_id,
            name="search_flights",
        )
    result = await _flights_agent.ainvoke(
        {"messages": [{"role": "user", "content": request}]},
        config={"recursion_limit": 6},
    )
    status = "error" if _child_had_error(result["messages"]) else "success"
    return ToolMessage(
        content=result["messages"][-1].content, status=status, tool_call_id=tool_call_id, name="search_flights"
    )


@tool
async def search_hotels(
    request: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    destination: Annotated[str | None, InjectedState("destination")] = None,
) -> ToolMessage:
    """Delegate to a specialized hotel-search agent. Pass the trip details you have so far
    (destination, dates, party size, budget) as `request`. It may return concrete hotel
    options, or a clarifying question if something critical is missing."""
    if not destination:
        return ToolMessage(
            content="Rejected: no destination confirmed yet — call confirm_destination first.",
            status="error",
            tool_call_id=tool_call_id,
            name="search_hotels",
        )
    result = await _hotels_agent.ainvoke(
        {"messages": [{"role": "user", "content": request}]},
        config={"recursion_limit": 6},
    )
    status = "error" if _child_had_error(result["messages"]) else "success"
    return ToolMessage(
        content=result["messages"][-1].content, status=status, tool_call_id=tool_call_id, name="search_hotels"
    )
