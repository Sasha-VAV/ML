import json
import asyncio
from dataclasses import dataclass
from pathlib import Path
import os
import uuid

from qdrant_client import AsyncQdrantClient, models
from fastembed import SparseTextEmbedding
from tqdm.asyncio import tqdm_asyncio

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
    topic = topic.replace(".txt", "")

    res: list[RetrievalChunk] = []
    for line in document.content.splitlines():
        timing, speaker, content = line.split(" | ")
        chunk = RetrievalChunk(
            id=str(uuid.uuid4()),
            year=year,
            meeting=meeting,
            topic=topic,
            speaker=speaker,
            content=content,
        )
        res.append(chunk)

    return res


class Retrieval:
    def __init__(self, settings: QdrantSettings, embedder: Embedder):
        self.client = AsyncQdrantClient(host=settings.host, port=settings.port)
        self.settings = settings
        self.embedder = embedder
        self.sparse_text_embedding_ru: SparseTextEmbedding | None = None

    def start(self, data: list[Document]):
        get_avg_doc_length = self.get_avg_document_length(data)
        print(f"Average document length: {get_avg_doc_length}")
        self.sparse_text_embedding_ru = SparseTextEmbedding(
            "Qdrant/bm25", language="russian", avg_doc_length=get_avg_doc_length
        )

        if not asyncio.run(
            self.client.collection_exists(self.settings.collection_name)
        ):
            asyncio.run(self._create_collection(self.settings.collection_name))
        if asyncio.run(self.client.count(self.settings.collection_name)).count == 0:
            asyncio.run(self._populate_collection(self.settings.collection_name, data))

    def get_avg_document_length(self, data: list[Document]) -> float:
        bm25 = SparseTextEmbedding("Qdrant/bm25", language="russian")
        lens = [
            len(next(bm25.embed([chunk.content])).indices)  # type: ignore
            for document in data
            for chunk in from_document(document)
        ]
        return sum(lens) / len(lens)

    async def _create_collection(self, collection_name: str):
        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=1536,  # MUST match your model
                    distance=models.Distance.COSINE,  # normalizes on write
                ),
            },
            sparse_vectors_config={
                "bm25_ru": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,  # <-- without this BM25 is silently wrong
                ),
            },
        )
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="year",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="meeting",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="topic",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="speaker",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name="content",
            field_schema=models.PayloadSchemaType.TEXT,
        )

    async def _populate_collection(self, collection_name: str, data: list[Document]):
        if self.sparse_text_embedding_ru is None:
            raise ValueError("Sparse text embeddings are not initialized.")

        chunks: list[RetrievalChunk] = []
        for document in data:
            chunks.extend(from_document(document))

        async def process_batch(batch: list[RetrievalChunk]):
            embedded = await asyncio.gather(
                *[self.embedder.embed([chunk.content]) for chunk in batch]
            )
            vectors = [v[0] for v in embedded]
            sparse_ru = await asyncio.to_thread(lambda: list(self.sparse_text_embedding_ru.embed([chunk.content for chunk in batch])))  # type: ignore
            await self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=chunk.id,
                        vector={
                            "dense": list(vector),
                            "bm25_ru": models.SparseVector(
                                indices=s_ru.indices.tolist(),
                                values=s_ru.values.tolist(),
                            ),
                        },
                        payload={
                            "year": chunk.year,
                            "meeting": chunk.meeting,
                            "topic": chunk.topic,
                            "speaker": chunk.speaker,
                            "content": chunk.content,
                        },
                    )
                    for chunk, vector, s_ru in zip(batch, vectors, sparse_ru)
                ],
                wait=False,
            )

        await tqdm_asyncio.gather(
            *[process_batch(chunks[i : i + 64]) for i in range(0, len(chunks), 64)],
            desc="Populating Qdrant collection",
        )

    async def query(
        self,
        queries: list[str],
        top_k: int,
        *,
        year: int | None = None,
        meeting: str | None = None,
        topic: str | None = None,
        speaker: str | None = None,
    ) -> str:
        top_k = min(top_k, 10)
        must_have_filters = []
        if year is not None:
            must_have_filters.append(
                models.FieldCondition(key="year", match=models.MatchValue(value=year))
            )
        if meeting is not None:
            must_have_filters.append(
                models.FieldCondition(
                    key="meeting", match=models.MatchValue(value=meeting)
                )
            )
        if topic is not None:
            must_have_filters.append(
                models.FieldCondition(key="topic", match=models.MatchValue(value=topic))
            )
        if speaker is not None:
            must_have_filters.append(
                models.FieldCondition(
                    key="speaker", match=models.MatchValue(value=speaker)
                )
            )

        flt = models.Filter(must=must_have_filters) if must_have_filters else None

        async def embed_query(query: str):
            dense_vector = await self.embedder.embed([query])
            sparse_vector = await asyncio.to_thread(lambda: list(self.sparse_text_embedding_ru.query_embed([query])))  # type: ignore
            return dense_vector[0], sparse_vector[0]

        results = await tqdm_asyncio.gather(*[embed_query(query) for query in queries])

        chunks = await self.client.query_points(
            self.settings.collection_name,
            prefetch=[
                p
                for dense, sparse in results
                for p in (
                    models.Prefetch(query=dense, using="dense", filter=flt, limit=50),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                        using="bm25_ru",
                        filter=flt,
                        limit=50,
                    ),
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return json.dumps([chunk.payload for chunk in chunks.points], indent=2, ensure_ascii=False) 

    async def discover(self, target: str, year: int | None = None,
            meeting: str | None = None,
            topic: str | None = None,
            speaker: str | None = None,) -> list[str]:

        must_have_filters = []
        if year is not None:
            must_have_filters.append(
                models.FieldCondition(key="year", match=models.MatchValue(value=year))
            )
        if meeting is not None:
            must_have_filters.append(
                models.FieldCondition(
                    key="meeting", match=models.MatchValue(value=meeting)
                )
            )
        if topic is not None:
            must_have_filters.append(
                models.FieldCondition(key="topic", match=models.MatchValue(value=topic))
            )
        if speaker is not None:
            must_have_filters.append(
                models.FieldCondition(
                    key="speaker", match=models.MatchValue(value=speaker)
                )
            )
        flt = models.Filter(must=must_have_filters) if must_have_filters else None

        resp = await self.client.facet(
            collection_name=self.settings.collection_name,
            key=target,
            facet_filter=flt,
            limit=100,
        )

        return [str(item.value) for item in resp.hits]
