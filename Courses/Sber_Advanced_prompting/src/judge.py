import uuid

from langchain.chat_models import BaseChatModel
from langchain_core.callbacks import UsageMetadataCallbackHandler
from pydantic import BaseModel, Field

from src.agent import get_agent
from src.context import AgentContext
from src.schemas import Answer


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
    """Scores a single answer against its reference answer.

    Args:
        model: Chat model used as the judge.
        question: The golden-set question.
        reference: The golden-set reference answer.
        answer: The answer produced by the pipeline.

    Returns:
        A `Verdict` with the judge's reasoning and its pass/fail decision.
    """
    judge = model.with_structured_output(Verdict)
    return await judge.ainvoke(
        JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    )


async def answer_question(
    model: BaseChatModel, context: AgentContext, question: str
) -> tuple[Answer | None, dict[str, int]]:
    """Runs the pipeline on one question in its own conversation thread.

    Token usage is captured per question so pipeline cost can be reported
    separately from the judge's own consumption.

    Args:
        model: Chat model backing the agent.
        context: Runtime dependencies (the Qdrant-backed retriever).
        question: The question to answer.

    Returns:
        The structured `Answer` (None if the model returned none) and a dict of
        `input_tokens` / `output_tokens` / `total_tokens` spent answering it.
    """
    usage_handler = UsageMetadataCallbackHandler()
    agent = get_agent(model)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config={
            "configurable": {"thread_id": str(uuid.uuid4())},
            "callbacks": [usage_handler],
        },
        context=context,
    )

    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for per_model in usage_handler.usage_metadata.values():
        for key in totals:
            totals[key] += per_model.get(key, 0)

    return result.get("structured_response"), totals
