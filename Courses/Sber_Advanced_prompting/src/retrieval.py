from dataclasses import dataclass
from pathlib import Path
import os
import uuid

from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding


from src.config import QdrantSettings
from src.models import Embedder


@dataclass
class Document:
    name: str
    content: str

def load_data(path: Path) -> list[Document]:
    res: list[Document] = []
    for file in os.listdir(path):
        with open(path / file, "r", encoding="utf-8") as f:
            content = f.read()
            res.append(Document(name=file, content=content))
    return res

@dataclass
class RetrievalChunk:
    id: str
    year: int
    meeting: str
    topic: str 
    speaker: str
    content: str

def from_document(document: Document) -> list[RetrievalChunk]:
    _, year, _, _, meeting, topic, *_ = document.name.split("_")
    year = int(year)

    res: list[RetrievalChunk] = []
    for line in document.content.splitlines():
        timing, speaker, content = line.split(" | ")
        chunk = RetrievalChunk(
            id=str(uuid.uuid4()),
            year=year,
            meeting=meeting,
            topic=topic,
            speaker=speaker,
            content=content
        )
        res.append(chunk)

    return res


class Retrieval:
    def __init__(self, settings: QdrantSettings, embedder: Embedder):
        self.client = QdrantClient(host=settings.host, port=settings.port)
        self.settings = settings
        self.embedder = embedder
        self.sparse_text_embedding_ru: SparseTextEmbedding | None = None
        self.sparse_text_embedding_en: SparseTextEmbedding | None = None

    def start(self, data: list[Document]):
        get_avg_doc_length = self.get_avg_document_length(data)
        print(f"Average document length: {get_avg_doc_length}")
        self.sparse_text_embedding_ru = SparseTextEmbedding("Qdrant/bm25", language="russian", avg_doc_length=get_avg_doc_length)
        self.sparse_text_embedding_en = SparseTextEmbedding("Qdrant/bm25", language="english", avg_doc_length=get_avg_doc_length)

        if not self.client.collection_exists(self.settings.collection_name):
            self._create_collection(self.settings.collection_name)
        if len(data) == 0 or len(data) != self.client.count(self.settings.collection_name).count:
            self._populate_collection(self.settings.collection_name, data)

    def get_avg_document_length(self, data: list[Document]) -> float:
        bm25 = SparseTextEmbedding("Qdrant/bm25", language="russian")
        lens = [
            len(next(bm25.embed([
                chunk.content
            ])).indices)
            for document in data
            for chunk in from_document(document)
        ]
        return sum(lens) / len(lens)

    def _create_collection(self, collection_name: str):
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=1536,                             # MUST match your model
                    distance=models.Distance.COSINE,      # normalizes on write
                ),
            },
            sparse_vectors_config={
                "bm25_en": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,         # <-- without this BM25 is silently wrong
                ),
                "bm25_ru": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,         # <-- without this BM25 is silently wrong
                ),
            },
        )

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="year",
            field_schema=models.PayloadSchemaType.INTEGER
        )

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="meeting",
            field_schema=models.PayloadSchemaType.KEYWORD
        )

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="topic",
            field_schema=models.PayloadSchemaType.KEYWORD
        )

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="speaker",
            field_schema=models.PayloadSchemaType.KEYWORD
        )

        self.client.create_payload_index(
            collection_name=collection_name,
            field_name="content",
            field_schema=models.PayloadSchemaType.TEXT
        )


    def _populate_collection(self, collection_name: str, data: list[Document]):
        chunks: list[RetrievalChunk] = []
        for document in data:
            chunks.extend(from_document(document))

        
        