from pydantic_settings import BaseSettings


class QdrantSettings(BaseSettings):
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "spief"


class EmbedderSettings(BaseSettings):
    endpoint: str = "http://212.109.220.252:8023/api/v1/embedder/predict"


class Settings(BaseSettings):
    qdrant: QdrantSettings = QdrantSettings()
    embedder: EmbedderSettings = EmbedderSettings()
