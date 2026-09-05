import asyncio
import uuid

from pydantic import BaseModel, Field
from langchain.chat_models import BaseChatModel

from src.agent import get_agent
from src.context import AgentContext


class Verdict(BaseModel):
    """One judged answer.

    Field order is load-bearing: structured output is generated in schema
    order, so `reason` comes first to make the verdict follow from the
    comparison rather than get rationalized after the fact.
    """

    reason: str = Field(
        description="One short sentence comparing the answer to the reference."
    )
    ok: bool = Field(description="True if the answer is good enough, False otherwise.")


JUDGE_PROMPT = """\
You grade an assistant's answer to a question about the SPIEF forum against a \
reference answer.

Mark it ok if it conveys the same facts as the reference - names, numbers and \
dates must match. Wording, extra correct detail, and a different level of \
verbosity are fine. Mark it not ok if it contradicts the reference, misses the \
facts the question asked for, or says it could not find anything.

Question:
{question}

Reference answer:
{reference}

Assistant's answer:
{answer}
"""


async def judge_answer(
    model: BaseChatModel, question: str, reference: str, answer: str
) -> Verdict:
    """Scores a single answer against its reference."""
    judge = model.with_structured_output(Verdict)
    return await judge.ainvoke(
        JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    )


async def answer_question(model: BaseChatModel, context: AgentContext, question: str) -> str:
    """Runs the agent on one question in its own thread."""
    agent = get_agent(model)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
        context=context,
    )
    return result["messages"][-1].content
