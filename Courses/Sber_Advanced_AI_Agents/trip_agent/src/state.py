from typing import NotRequired

from langchain.agents import AgentState


class TripState(AgentState):
    active_skill: NotRequired[str | None]
    destination: NotRequired[str | None]
    flight: NotRequired[str | None]
    hotel: NotRequired[str | None]
