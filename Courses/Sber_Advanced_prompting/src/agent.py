from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from src.context import AgentContext
from src.middleware import EvidenceMiddleware
from src.prompts import SYSTEM_PROMPT
from src.schemas import Answer
from src.state import QAAgentState
from src.tools import tools


def get_agent(model: BaseChatModel, checkpointer: BaseCheckpointSaver | None = None):
    """Builds the SPIEF question-answering agent.

    The agent searches the transcript corpus with the tools in `src.tools` and
    returns its final answer in the fixed `Answer` schema, exposed on the result
    as `structured_response`.

    Args:
        model: Chat model used for both reasoning and tool calling.
        checkpointer: Conversation store; an in-memory one is used if omitted.

    Returns:
        A compiled LangGraph agent, invoked with `context=AgentContext(...)`.
    """
    return create_agent(
        model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        state_schema=QAAgentState,
        context_schema=AgentContext,
        response_format=ToolStrategy(Answer),
        middleware=[EvidenceMiddleware()],
        checkpointer=checkpointer or InMemorySaver(),
    )
