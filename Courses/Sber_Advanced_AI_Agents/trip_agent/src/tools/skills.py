from typing import Literal

from langchain_core.tools import tool

from src.skill import load_all_skills

SKILLS = load_all_skills()


def skills_summary() -> str:
    return "\n".join(f"- {name}: {skill.description}" for name, skill in SKILLS.items())


@tool
def load_skill_content(skill_name: Literal["destination-advisor", "trip-booking"]) -> str:
    """Load the full instructions for one skill so you can follow them.

    Call this once you've decided a skill applies to the user's request, then
    follow its instructions for the rest of your reply.
    """
    return SKILLS[skill_name].system_prompt
