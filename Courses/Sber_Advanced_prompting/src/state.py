import operator
from typing import Annotated, TypedDict

from langchain.agents.middleware import AgentState


class Evidence(TypedDict):
    """One fact the model chose to remember, separate from chat history."""

    id: str
    content: str


class QAAgentState(AgentState):
    """Agent state with a durable evidence store alongside the message list."""

    evidence: Annotated[list[Evidence], operator.add]
