from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from src.state import QAAgentState as EvidenceState

# Tool name whose raw output is bulky retrieved text: dropped from history
# once the model has had one turn to read it. The model decides what's worth
# keeping by calling `save_facts` - this middleware never invents evidence.
RAW_RETRIEVAL_TOOL = "retrieve_data"

# How many of the most recent tool-call rounds keep the raw retrieval text.
KEEP_FULL_ROUNDS = 1

PLACEHOLDER = (
    "[dropped to save tokens - call save_facts right after retrieve_data next "
    "time if you need to keep something from it; call retrieve_data again for "
    "the raw text]"
)


class EvidenceMiddleware(AgentMiddleware[EvidenceState]):
    """Drops raw `retrieve_data` output from history after one round.

    The model is expected to call `save_facts` with whatever it needs to
    remember before the raw retrieval result ages out of the transcript.
    Saved facts live in `state["evidence"]` and are surfaced back to the
    model as a digest before every model call.
    """

    state_schema = EvidenceState

    def before_model(self, state: EvidenceState, runtime: Any) -> dict[str, Any] | None:
        """Replaces aged-out raw retrieval results with a short placeholder.

        Args:
            state: Current agent state.
            runtime: LangGraph runtime (unused).

        Returns:
            A message update, or None when nothing needed dropping.
        """
        messages = state["messages"]

        round_idx = 0
        round_of: dict[str, int] = {}
        tool_name_of: dict[str, str] = {}
        for message in messages:
            if isinstance(message, AIMessage) and message.tool_calls:
                round_idx += 1
                for call in message.tool_calls:
                    if call["id"] is not None:
                        tool_name_of[call["id"]] = call["name"]
            elif isinstance(message, ToolMessage):
                round_of[message.tool_call_id] = round_idx
        total_rounds = round_idx

        updates = []
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            if tool_name_of.get(message.tool_call_id) != RAW_RETRIEVAL_TOOL:
                continue
            if total_rounds - round_of[message.tool_call_id] < KEEP_FULL_ROUNDS:
                continue  # part of a recent-enough round: leave it full
            if message.content == PLACEHOLDER:
                continue  # already dropped
            updates.append(message.model_copy(update={"content": PLACEHOLDER}))

        return {"messages": updates} if updates else None

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Prepends a digest of saved facts to every model call.

        Args:
            request: The outgoing model request.
            handler: Next handler in the middleware chain.

        Returns:
            The model response produced by the handler.
        """
        evidence = request.state.get("evidence", [])
        if not evidence:
            return await handler(request)

        lines = (f"- [{e['id']}] {e['content']}" for e in evidence)
        digest = SystemMessage(
            content="Facts saved so far via save_facts:\n" + "\n".join(lines)
        )
        return await handler(request.override(messages=[digest, *request.messages]))
