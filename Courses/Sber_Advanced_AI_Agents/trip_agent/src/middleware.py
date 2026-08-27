from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import ToolMessage
from langfuse import get_client
from langgraph.types import Command

_ALWAYS_AVAILABLE = {"load_destination_advisor_skill"}

_SKILL_TOOLS = {
    "destination-advisor": {"confirm_destination"},
    "trip-booking": {"search_flights", "search_hotels", "confirm_flight", "confirm_hotel", "book_trip"},
}


class SkillScopedToolsMiddleware(AgentMiddleware):
    """Only exposes a skill's tools once that skill has been loaded, and only exposes
    load_trip_booking_skill once a destination has been confirmed."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        state = request.state
        active_skill = state.get("active_skill")
        allowed = set(_ALWAYS_AVAILABLE) | _SKILL_TOOLS.get(active_skill, set())
        if state.get("destination"):
            allowed.add("load_trip_booking_skill")

        request = request.override(
            tools=[t for t in request.tools if getattr(t, "name", None) in allowed]
        )
        return await handler(request)


class BookingReadinessMiddleware(AgentMiddleware):
    """Hides book_trip from the model's tool schema until flight and hotel are confirmed."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        state = request.state
        # If either flight or hotel is not confirmed, remove book_trip from the tools list
        if not (state.get("flight") and state.get("hotel")):
            request = request.override(
                tools=[t for t in request.tools if getattr(t, "name", None) != "book_trip"]
            )
        return await handler(request)


class ToolMetricsMiddleware(AgentMiddleware):
    """Reports tool_call_count/tool_success to Langfuse at the point each tool actually
    executes, instead of reconstructing it after the fact from the message list."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        result = await handler(request)

        message = result.update["messages"][0] if isinstance(result, Command) else result
        success = not (isinstance(message, ToolMessage) and message.status == "error")

        client = get_client()
        tool_name = request.tool_call.get("name")
        client.score_current_trace(name="tool_call_count", value=1, data_type="NUMERIC", comment=tool_name)
        client.score_current_trace(
            name="tool_success", value=1.0 if success else 0.0, data_type="NUMERIC", comment=tool_name
        )
        return result
