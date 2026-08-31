import uuid
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from src.queue import BookingTask, default_queue


@tool
async def confirm_flight(flight: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Record the specific flight option the traveler has chosen from search_flights results."""
    return Command(
        update={
            "flight": flight,
            "messages": [ToolMessage(content=f"Flight confirmed: {flight}.", tool_call_id=tool_call_id)],
        }
    )


@tool
async def confirm_hotel(hotel: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Record the specific hotel option the traveler has chosen from search_hotels results."""
    return Command(
        update={
            "hotel": hotel,
            "messages": [ToolMessage(content=f"Hotel confirmed: {hotel}.", tool_call_id=tool_call_id)],
        }
    )


@tool
async def book_trip(
    dates: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    destination: Annotated[str | None, InjectedState("destination")] = None,
    flight: Annotated[str | None, InjectedState("flight")] = None,
    hotel: Annotated[str | None, InjectedState("hotel")] = None,
) -> ToolMessage:
    """Finalize a booking. Requires destination, flight, and hotel to already be confirmed
    via confirm_destination/confirm_flight/confirm_hotel — this tool trusts recorded state,
    not whatever you say in the call. Goes through the booking queue (publish/reserve/save/
    ack, bounded retry, DLQ) and returns the outcome."""
    missing = [name for name, value in (("destination", destination), ("flight", flight), ("hotel", hotel)) if not value]
    if missing:
        return ToolMessage(
            content=f"Rejected: cannot book yet — missing confirmed {', '.join(missing)}.",
            status="error",
            tool_call_id=tool_call_id,
            name="book_trip",
        )

    payload = f"Booking {destination}, {dates}. Flight: {flight}. Hotel: {hotel}. Answer with either: y/N. \n -> "
    if input(payload).strip().lower() != "y":
        return ToolMessage(content="Booking canceled by user.", tool_call_id=tool_call_id, name="book_trip")

    idempotency_key = f"{destination}|{flight}|{hotel}"
    task = BookingTask(
        message_id=str(uuid.uuid4()),
        idempotency_key=idempotency_key,
        destination=destination,
        dates=dates,
        flight=flight,
        hotel=hotel,
    )
    default_queue.publish(task)

    for _ in range(task.max_attempts + 1):
        if default_queue.is_acked(idempotency_key) or default_queue.dlq_reason(idempotency_key):
            break
        await default_queue.run_worker_once()

    confirmation = default_queue.result_for(idempotency_key)
    if confirmation:
        return ToolMessage(
            content=(
                f"Booked: {destination}, {dates}. Flight: {flight}. Hotel: {hotel}. "
                f"Confirmation: {confirmation}"
            ),
            tool_call_id=tool_call_id,
            name="book_trip",
        )

    reason = default_queue.dlq_reason(idempotency_key)
    return ToolMessage(
        content=f"Booking failed after retries and was moved to the dead-letter queue ({reason}). Tell the traveler booking is temporarily unavailable.",
        status="error",
        tool_call_id=tool_call_id,
        name="book_trip",
    )
