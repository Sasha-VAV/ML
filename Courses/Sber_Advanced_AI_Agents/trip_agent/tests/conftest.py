import pytest
from dotenv import load_dotenv

from src import providers

load_dotenv()


@pytest.fixture(autouse=True)
def _isolate_providers():
    """Every test installs whatever providers it needs; restore the originals after."""
    original = (providers.flight_provider, providers.hotel_provider, providers.booking_provider)
    yield
    providers.flight_provider, providers.hotel_provider, providers.booking_provider = original
