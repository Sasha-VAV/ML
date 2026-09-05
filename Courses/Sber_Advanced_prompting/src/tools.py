import json
import uuid
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langchain.tools import ToolRuntime
from langgraph.types import Command

from src.context import AgentContext
from src.retrieval import FacetField
from src.state import Evidence, QAAgentState

Runtime = ToolRuntime[AgentContext, QAAgentState]


async def discover_values(
    field: FacetField,
    runtime: Runtime,
    year: int | None = None,
    meeting: str | None = None,
    topic: str | None = None,
    speaker: str | None = None,
) -> str:
    """
    Lists the values that actually exist for one metadata field of the SPIEF corpus.

    Use this before `retrieve_data` whenever you intend to filter: it tells you
    the exact spelling of a speaker, meeting or topic, so the filter matches
    something instead of silently returning nothing. The optional arguments
    narrow the slice being described (e.g. field="speaker", year=2012 lists the
    speakers of 2012 only).

    Args:
        field (str): Field to list values of - one of "year", "meeting", "topic", "speaker".
        year (int, optional): Restrict to this year of the SPIEF. Defaults to None.
        meeting (str, optional): Restrict to this meeting. Defaults to None.
        topic (str, optional): Restrict to this topic. Defaults to None.
        speaker (str, optional): Restrict to this speaker. Defaults to None.

    Returns:
        str: JSON list of {"value", "count"} objects, ordered by descending count.

    Example:
        >>> await discover_values("speaker", year=2012)
        '[{"value": "Владимир Путин", "count": 42}, ...]'
    """
    hits = await runtime.context.retrieval.discover(
        field,
        year=year,
        meeting=meeting,
        topic=topic,
        speaker=speaker,
    )
    return json.dumps(hits, indent=2, ensure_ascii=False)


async def retrieve_data(
    queries: list[str],
    runtime: Runtime,
    top_k: int = 5,
    year: int | None = None,
    meeting: str | None = None,
    topic: str | None = None,
    speaker: str | None = None,
) -> str:
    """
    Retrieves transcript excerpts of SPIEF meetings by hybrid (dense + BM25) search.

    Filter arguments are exact matches: get their spelling from `discover_values`
    first rather than guessing, since a value that does not exist yields nothing.

    Args:
        queries (list[str]): Search queries; results across all of them are fused.
        top_k (int, optional): Number of excerpts to return for **all** queries. Capped at 10. Defaults to 5.
        year (int, optional): Only excerpts from this year of the SPIEF. Defaults to None.
        meeting (str, optional): Only excerpts from this meeting. Defaults to None.
        topic (str, optional): Only excerpts from this topic. Defaults to None.
        speaker (str, optional): Only excerpts spoken by this speaker. Defaults to None.

    Returns:
        str: JSON list of excerpts, each with its year, meeting, topic, speaker and content.
    """
    return await runtime.context.retrieval.query(
        queries=queries,
        top_k=top_k,
        year=year,
        meeting=meeting,
        topic=topic,
        speaker=speaker,
    )


def save_facts(
    facts: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Saves the key facts you want to remember from data you just retrieved.

    Call this right after `retrieve_data` with the specific statements you
    will need to answer the question (quotes, numbers, names, dates). The raw
    `retrieve_data` result is dropped from context on the next step to save
    tokens, so anything worth keeping has to be written down here first.

    Args:
        facts (list[str]): Short, self-contained factual statements, each usable on its own.

    Returns:
        Command: Appends the given facts to the agent's evidence store.
    """
    entries: list[Evidence] = [
        Evidence(id=uuid.uuid4().hex[:8], content=fact) for fact in facts
    ]
    return Command(
        update={
            "evidence": entries,
            "messages": [
                ToolMessage(
                    content=f"Saved {len(facts)} fact(s).", tool_call_id=tool_call_id
                )
            ],
        }
    )


tools = [
    discover_values,
    retrieve_data,
    save_facts,
]
