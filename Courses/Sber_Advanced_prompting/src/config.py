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

    # Cap on how many transcript files get indexed; None ingests the whole corpus.
    # Keep it small for a quick local run, unset it for real answers.
    max_documents: int | None = None
