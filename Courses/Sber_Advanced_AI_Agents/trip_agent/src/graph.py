from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from src.llm import get_agent_llm
from src.middleware import BookingReadinessMiddleware, SkillScopedToolsMiddleware, ToolMetricsMiddleware
from src.state import TripState
from src.tools import (
    book_trip,
    confirm_destination,
    confirm_flight,
    confirm_hotel,
    load_destination_advisor_skill,
    load_trip_booking_skill,
    search_flights,
    search_hotels,
    skills_summary,
)

SYSTEM_PROMPT = f"""You are a trip-planning assistant.

You have access to specialized skills, each with detailed instructions for a
specific kind of request. Skills available:

{skills_summary()}

Look at the tools you actually have available right now, not just this
description — call whichever load_*_skill tool is present and matches what
the traveler needs, then follow that skill's instructions for the rest of
this reply. A skill's tool only appears once its precondition is met (e.g.
booking needs a destination confirmed first), so if you don't see a tool you
expected, that means its precondition isn't met yet — go work toward it
instead of describing the situation or waiting.

If nothing you have fits the request (small talk, general questions,
anything outside trip planning), just answer directly and helpfully.

A skill's other tools only appear once you've loaded that skill. Some tools
enforce ordering and will reject a call with a reason instead of acting —
read the rejection and follow what it asks for instead of retrying the same
call.

Act as soon as you have what a tool needs — don't ask the traveler a
separate "should I proceed?" or "shall I confirm that?" question first.
Tools named confirm_* record a choice the traveler already stated in the
conversation; they are not a request for the traveler to confirm again. If
the traveler already told you the answer, just call the tool."""


def build_graph():
    return create_agent(
        get_agent_llm(),
        tools=[
            load_destination_advisor_skill,
            load_trip_booking_skill,
            confirm_destination,
            search_flights,
            search_hotels,
            confirm_flight,
            confirm_hotel,
            book_trip,
        ],
        system_prompt=SYSTEM_PROMPT,
        state_schema=TripState,
        middleware=[SkillScopedToolsMiddleware(), BookingReadinessMiddleware(), ToolMetricsMiddleware()],
        checkpointer=MemorySaver(),
    )
