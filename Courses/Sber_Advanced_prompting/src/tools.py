import uuid
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command

from src.state import Evidence


async def get_tags(
    year: int,
) -> list[str]:
    """
    Fetches the tags associated with the SPIEF event for a given year to use as metadata in rag search

    Args:
        year (int): The year of the SPIEF event
    
    Returns:
        list[str]: A list of tags associated with the SPIEF event for the given year.
    
    Example:
        >>> tags = await get_tags(2010)
        >>> print(tags)
        ['it', 'economy', 'vr']
    """
    ...


async def retrieve_data(
    queries: list[str],
    top_k: int = 5,
    year: int | None = None,
    tags: list[str] | None = None,
) -> str:
    """
    Retrieves transcripts of all meetings on the SPIEF

    Args:
        queries (list[str]): A list of queries to search for in the transcripts.
        top_k (int, optional): The number of top results to return for **all** queries.
        year (int, optional): The year of the SPIEF to filter the transcripts. Defaults to None.
        tags (list[str], optional): A list of tags to filter the transcripts. Gives higher score to transcripts with at least one of matching tags. Defaults to None.

    Returns:
        str: A string combined transcripts of some meetings that occurred in the SPIEF
    """
    ...


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
                ToolMessage(content=f"Saved {len(facts)} fact(s).", tool_call_id=tool_call_id)
            ],
        }
    )


tools = [
    get_tags,
    retrieve_data,
    save_facts,
]