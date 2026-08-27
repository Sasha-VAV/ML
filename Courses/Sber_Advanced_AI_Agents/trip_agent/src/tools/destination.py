from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command


@tool
async def confirm_destination(destination: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Lock in the traveler's destination once they've agreed on one."""
    return Command(
        update={
            "destination": destination,
            "messages": [ToolMessage(content=f"Destination confirmed: {destination}.", tool_call_id=tool_call_id)],
        }
    )
