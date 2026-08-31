from langchain_core.tools import tool

from src import providers


@tool
async def find_tickets(destination: str, depart_date: str, return_date: str, party_size: int = 1) -> str:
    """Mock flight search. Returns 2-3 plausible round-trip flight options (airline, total
    price, flight duration) to `destination` between `depart_date` and `return_date`."""
    return await providers.flight_provider.search(destination, depart_date, return_date, party_size)


@tool
async def find_hotel(destination: str, check_in: str, check_out: str, party_size: int = 1) -> str:
    """Mock hotel search. Returns 2 plausible hotel options (name, price/night, rating) in
    `destination` between `check_in` and `check_out`."""
    return await providers.hotel_provider.search(destination, check_in, check_out, party_size)
