import aiohttp
import asyncio


from src.config import EmbedderSettings


class Embedder:
    """Client for the external dense-embedding service.

    Concurrency is capped with a semaphore so bulk indexing cannot overwhelm
    the endpoint.
    """

    def __init__(self, settings: EmbedderSettings):
        self.settings = settings
        self.semaphore = asyncio.Semaphore(8)  # Limit concurrent requests to 8

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeds texts, waiting for a free concurrency slot.

        Args:
            texts: Texts to embed.

        Returns:
            One dense vector per input text.
        """
        async with self.semaphore:
            while True:
                try:
                    return await self._embed(texts)
                except Exception as e:
                    print(f"Error during embedding: {e}. Retrying in 5 seconds...")
                    await asyncio.sleep(5)
            

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.settings.endpoint,
                json={"query": texts},
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                return response_data["data"]
