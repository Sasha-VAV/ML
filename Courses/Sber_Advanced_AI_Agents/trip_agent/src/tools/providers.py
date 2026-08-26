import random

from langchain_core.tools import tool

_AIRLINES = ["AeroLumen", "Northwind Air", "BlueArc Airways", "Continental Skyline"]
_HOTEL_NAMES = ["Harbor View Inn", "The Old Quarter Hotel", "Meridian Suites", "Garden Court Hotel"]


@tool
def find_tickets(destination: str, depart_date: str, return_date: str, party_size: int = 1) -> str:
    """Mock flight search. Returns 2-3 plausible round-trip flight options (airline, total
    price, flight duration) to `destination` between `depart_date` and `return_date`."""
    options = [
        f"{airline}: ${random.randint(180, 850) * party_size} total, ~{random.randint(2, 14)}h flight time"
        for airline in random.sample(_AIRLINES, k=3)
    ]
    return "\n".join(options)


@tool
def find_hotel(destination: str, check_in: str, check_out: str, party_size: int = 1) -> str:
    """Mock hotel search. Returns 2 plausible hotel options (name, price/night, rating) in
    `destination` between `check_in` and `check_out`."""
    options = [
        f"{name}: ${random.randint(60, 320)}/night, {round(random.uniform(3.5, 4.9), 1)}★"
        for name in random.sample(_HOTEL_NAMES, k=2)
    ]
    return "\n".join(options)
