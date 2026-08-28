"""The 5 required demo scenarios, run against the real GigaChat endpoint (via the
local gpt2giga proxy) wherever the model is actually needed. Pure-logic pieces
(trust boundary, queue idempotency/DLQ) are tested directly with no LLM call."""

import uuid

from langchain_core.messages import ToolMessage

from src import providers
from src.graph import build_graph
from src.middleware import BookingReadinessMiddleware, SkillScopedToolsMiddleware
from src.providers import ErrorFlightProvider, MockBookingProvider, MockFlightProvider, MockHotelProvider, SingleErrorHotelProvider
from src.queue import BookingQueue, BookingTask
from src.tools import book_trip, load_destination_advisor_skill, load_trip_booking_skill, search_flights


def _new_config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


# --- Scenario 1: happy long route ------------------------------------------------
# Several dependent reads (search_flights, search_hotels), a skill switch, and a
# child agent, ending in a verified booking.

async def test_happy_long_route(monkeypatch):
    providers.flight_provider = MockFlightProvider()
    providers.hotel_provider = MockHotelProvider()
    providers.booking_provider = MockBookingProvider()
    monkeypatch.setattr("builtins.input", lambda _: "y")

    graph = build_graph()
    config = _new_config()

    async def send(text: str) -> dict:
        return await graph.ainvoke({"messages": [{"role": "user", "content": text}]}, config=config)

    await send("I'm not sure where to go, I like warm beaches on a budget")
    state = await send("Actually let's go with Lisbon")
    assert state["destination"]

    await send(
        "Let's book it. Dates Oct 3 to Oct 10, 2 travelers, budget $2000. "
        "Show me flight and hotel options."
    )
    state = await send(
        "I'll take the cheapest flight option and the cheapest hotel option — "
        "please confirm both and book the trip with those exact dates."
    )

    assert state.get("flight")
    assert state.get("hotel")
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    tool_names = {m.name for m in tool_messages}
    assert "search_flights" in tool_names
    assert "search_hotels" in tool_names
    assert "book_trip" in tool_names


# --- Scenario 2: cheap/short route -----------------------------------------------
# A request outside trip-planning never loads a skill or calls a tool — visibly
# fewer model/tool calls than scenario 1's route.

async def test_cheap_route_skips_tools_entirely():
    graph = build_graph()
    config = _new_config()

    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "Hi, just testing you out — no trip planning needed."}]},
        config=config,
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages == []


# --- Scenario 3: rejected proposal ------------------------------------------------
# Unconfirmed effect, invalid transition, and bad state all get blocked before
# any real effect executes, each with a clear reason.

async def test_book_trip_rejects_when_state_incomplete():
    result = await book_trip.coroutine(dates="Oct 3-10", tool_call_id="tc1", destination=None, flight=None, hotel=None)
    assert result.status == "error"
    assert "missing confirmed" in result.content


async def test_book_trip_blocked_without_human_confirmation(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    result = await book_trip.coroutine(
        dates="Oct 3-10", tool_call_id="tc1", destination="Lisbon", flight="AeroLumen $400", hotel="Harbor View Inn"
    )
    assert "canceled" in result.content.lower()


async def test_invalid_transition_hides_tools_structurally():
    class _FakeRequest:
        def __init__(self, state, tools):
            self.state = state
            self.tools = tools

        def override(self, tools):
            self.tools = tools
            return self

    async def handler(request):
        return request

    skill_gate = SkillScopedToolsMiddleware()
    request = _FakeRequest(
        state={},  # no skill loaded, no destination confirmed
        tools=[load_destination_advisor_skill, load_trip_booking_skill, search_flights, book_trip],
    )
    result = await skill_gate.awrap_model_call(request, handler)
    assert {getattr(t, "name", None) for t in result.tools} == {"load_destination_advisor_skill"}

    booking_gate = BookingReadinessMiddleware()
    request = _FakeRequest(state={"flight": None, "hotel": None}, tools=[book_trip])
    result = await booking_gate.awrap_model_call(request, handler)
    assert result.tools == []


# --- Scenario 4: transient failure ------------------------------------------------
# A read fails, retries within a bounded limit, then either succeeds (from cache
# on a repeat call) or degrades gracefully once retries are exhausted.

async def test_transient_read_failure_then_success_and_cache():
    providers.hotel_provider = SingleErrorHotelProvider()

    from src.tools.search_agents import search_hotels

    first = await search_hotels.coroutine(
        request="Hotels in Lisbon, Oct 3-10, 2 guests", tool_call_id="tc1", destination="Lisbon"
    )
    assert first.status == "success"
    assert len(providers.hotel_provider._cache) == 1

    second = await search_hotels.coroutine(
        request="Hotels in Lisbon, Oct 3-10, 2 guests", tool_call_id="tc2", destination="Lisbon"
    )
    assert second.status == "success"


async def test_permanent_read_failure_degrades_gracefully():
    providers.flight_provider = ErrorFlightProvider()

    result = await search_flights.coroutine(
        request="Flights to Lisbon, Oct 3-10, 1 passenger", tool_call_id="tc1", destination="Lisbon"
    )
    assert result.status == "error"


# --- Scenario 5: redelivery after crash + retry exhaustion -----------------------
# publish -> reserve -> save -> ack are separately callable, so a crash between
# save and ack is simulated directly; a redelivered message must not repeat the
# business operation. Separately: exhausted retries land in the DLQ.

async def test_redelivery_after_crash_does_not_repeat_business_op():
    queue = BookingQueue()
    calls = []

    class CountingBookingProvider:
        async def reserve(self, destination, dates, flight, hotel):
            calls.append((destination, dates, flight, hotel))
            return "TRIP-FIXED"

    providers.booking_provider = CountingBookingProvider()

    key = "Lisbon|AeroLumen|Harbor View Inn"
    task = BookingTask(message_id="m1", idempotency_key=key, destination="Lisbon", dates="Oct 3-10", flight="AeroLumen", hotel="Harbor View Inn")

    confirmation = await queue._reserve(task)
    queue._save_result(task, confirmation)
    # Crash happens here — before _ack.

    assert not queue.is_acked(key)
    assert len(calls) == 1

    redelivered = BookingTask(message_id="m2-redelivered", idempotency_key=key, destination="Lisbon", dates="Oct 3-10", flight="AeroLumen", hotel="Harbor View Inn")
    queue.publish(redelivered)
    await queue.run_worker_once()

    assert queue.is_acked(key)
    assert queue.result_for(key) == "TRIP-FIXED"
    assert len(calls) == 1  # provider was not called again


async def test_retry_exhaustion_moves_task_to_dlq():
    queue = BookingQueue()

    class AlwaysFailingProvider:
        async def reserve(self, *_args, **_kwargs):
            raise TimeoutError("provider down")

    providers.booking_provider = AlwaysFailingProvider()

    key = "Prague|BlueArc|Meridian Suites"
    task = BookingTask(message_id="m1", idempotency_key=key, destination="Prague", dates="Nov 1-5", flight="BlueArc", hotel="Meridian Suites", max_attempts=3)
    queue.publish(task)

    for _ in range(task.max_attempts):
        await queue.run_worker_once()

    assert queue.dlq_reason(key) is not None
    assert queue.result_for(key) is None
