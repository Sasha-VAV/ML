from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

from src.skill import load_all_skills

SKILLS = load_all_skills()


def skills_summary() -> str:
    return "\n".join(f"- {name}: {skill.description}" for name, skill in SKILLS.items())


def _load_skill_command(skill_name: str, tool_call_id: str) -> Command:
    skill = SKILLS[skill_name]
    return Command(
        update={
            "active_skill": skill_name,
            "messages": [ToolMessage(content=skill.system_prompt, tool_call_id=tool_call_id)],
        }
    )


@tool
async def load_destination_advisor_skill(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Load the destination-advisor skill's instructions and switch into its tools for the
    rest of this conversation, until you load a different skill."""
    return _load_skill_command("destination-advisor", tool_call_id)


@tool
async def load_trip_booking_skill(tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Load the trip-booking skill's instructions and switch into its tools for the rest
    of this conversation, until you load a different skill."""
    return _load_skill_command("trip-booking", tool_call_id)
