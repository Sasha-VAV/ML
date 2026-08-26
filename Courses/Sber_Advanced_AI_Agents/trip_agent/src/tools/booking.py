import uuid

from langchain_core.tools import tool


@tool
def book_trip(destination: str, dates: str, flight: str, hotel: str) -> str:
    """Finalize a booking once destination, dates, a chosen flight, and a chosen hotel are all
    known and confirmed by the user. Returns a mock booking confirmation."""
    confirmation = f"TRIP-{uuid.uuid4().hex[:8].upper()}"
    return (
        f"Booked: {destination}, {dates}. Flight: {flight}. Hotel: {hotel}. "
        f"Confirmation: {confirmation}"
    )
