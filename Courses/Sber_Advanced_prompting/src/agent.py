from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel

from src.tools import tools
from src.prompts import SYSTEM_PROMPT
from src.state import QAAgentState
from src.middleware import EvidenceMiddleware


def get_agent(model: BaseChatModel):
    return create_agent(
        model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        state_schema=QAAgentState,
        middleware=[EvidenceMiddleware()],
    )
