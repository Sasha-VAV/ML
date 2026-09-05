import aiohttp
import asyncio


from src.config import EmbedderSettings


class Embedder:
    def __init__(self, settings: EmbedderSettings):
        self.settings = settings
        self.semaphore = asyncio.Semaphore(8)  # Limit concurrent requests to 8

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with self.semaphore:
            return await self._embed(texts)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.settings.endpoint,
                json={"query": texts},
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                return response_data["data"]
