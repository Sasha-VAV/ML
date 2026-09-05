import aiohttp


from src.config import EmbedderSettings


class Embedder:
    def __init__(self, settings: EmbedderSettings):
        self.settings = settings

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.settings.endpoint,
                json={"query": texts},
            ) as response:
                response.raise_for_status()
                response_data = await response.json()
                return response_data["data"]
