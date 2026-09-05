from dataclasses import dataclass

from src.retrieval import Retrieval


@dataclass
class AgentContext:
    """Per-run dependencies, injected into tools as `ToolRuntime.context`.

    Declared to `create_agent` as its `context_schema` and passed to
    `ainvoke(..., context=AgentContext(...))`, so tools reach the Qdrant-backed
    retriever through the framework rather than a module-level global.
    """

    retrieval: Retrieval
