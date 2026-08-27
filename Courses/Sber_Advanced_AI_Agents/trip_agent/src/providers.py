import asyncio
import random
import uuid
from typing import Protocol

_AIRLINES = ["AeroLumen", "Northwind Air", "BlueArc Airways", "Continental Skyline"]
_HOTEL_NAMES = ["Harbor View Inn", "The Old Quarter Hotel", "Meridian Suites", "Garden Court Hotel"]


class FlightProvider(Protocol):
    async def search(self, destination: str, depart_date: str, return_date: str, party_size: int) -> str: ...


class HotelProvider(Protocol):
    async def search(self, destination: str, check_in: str, check_out: str, party_size: int) -> str: ...


class MockFlightProvider:
    async def search(self, destination: str, depart_date: str, return_date: str, party_size: int = 1) -> str:
        options = [
            f"{airline}: ${random.randint(180, 850) * party_size} total, ~{random.randint(2, 14)}h flight time"
            for airline in random.sample(_AIRLINES, k=3)
        ]
        return "\n".join(options)


class MockHotelProvider:
    async def search(self, destination: str, check_in: str, check_out: str, party_size: int = 1) -> str:
        options = [
            f"{name}: ${random.randint(60, 320)}/night, {round(random.uniform(3.5, 4.9), 1)}★"
            for name in random.sample(_HOTEL_NAMES, k=2)
        ]
        return "\n".join(options)


class ErrorFlightProvider:
    """Always fails, to simulate a provider that's permanently down."""

    async def search(self, destination: str, depart_date: str, return_date: str, party_size: int = 1) -> str:
        await asyncio.sleep(0.5)
        raise TimeoutError("Resource was busy, ask again")


class SingleErrorHotelProvider:
    """Fails once per destination, then succeeds — and caches the successful result only."""

    def __init__(self):
        self._failed_once: set[str] = set()
        self._cache: dict[str, str] = {}

    async def search(self, destination: str, check_in: str, check_out: str, party_size: int = 1) -> str:
        if destination in self._cache:
            return self._cache[destination]

        if destination not in self._failed_once:
            self._failed_once.add(destination)
            await asyncio.sleep(0.5)
            raise TimeoutError("Resource was busy, ask again")

        options = [
            f"{name}: ${random.randint(60, 320)}/night, {round(random.uniform(3.5, 4.9), 1)}★"
            for name in random.sample(_HOTEL_NAMES, k=2)
        ]
        result = "\n".join(options)
        self._cache[destination] = result
        return result


class BookingProvider(Protocol):
    async def reserve(self, destination: str, dates: str, flight: str, hotel: str) -> str: ...


class MockBookingProvider:
    async def reserve(self, destination: str, dates: str, flight: str, hotel: str) -> str:
        return f"TRIP-{uuid.uuid4().hex[:8].upper()}"


flight_provider: FlightProvider = ErrorFlightProvider()
hotel_provider: HotelProvider = SingleErrorHotelProvider()
booking_provider: BookingProvider = MockBookingProvider()
