import json
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args
import os
import uuid

from qdrant_client import AsyncQdrantClient, models
from fastembed import SparseTextEmbedding
from tqdm.asyncio import tqdm_asyncio

from src.config import QdrantSettings
from src.models import Embedder


@dataclass
class Document:
    """One transcript file: its filename and full text."""

    name: str
    content: str


def load_data(path: Path) -> list[Document]:
    """Reads every transcript file in a directory.

    Args:
        path: Directory holding the SPIEF `.txt` transcripts.

    Returns:
        One `Document` per file.
    """
    res: list[Document] = []
    for file in os.listdir(path):
        with open(path / file, "r", encoding="utf-8") as f:
            content = f.read()
            res.append(Document(name=file, content=content))
    return res


# Payload fields that carry a keyword/integer index in Qdrant, and are therefore
# both filterable and facetable. `content` is a TEXT index: searchable, not facetable.
FacetField = Literal["year", "meeting", "topic", "speaker"]
FACET_FIELDS: tuple[str, ...] = get_args(FacetField)


@dataclass
class RetrievalChunk:
    """A single speaker turn, the unit that gets indexed and retrieved."""

    id: str
    year: int
    meeting: str
    topic: str
    speaker: str
    content: str


def from_document(document: Document) -> list[RetrievalChunk]:
    """Splits a transcript into per-speaker chunks.

    Metadata (year, meeting, topic) is parsed from the filename; the speaker
    and text come from each `timing | speaker | content` line.

    Args:
        document: The transcript to split.

    Returns:
        One chunk per speaker turn.
    """
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
    """Hybrid search over the SPIEF transcripts, backed by Qdrant.

    Combines dense vectors with a Russian BM25 sparse index, fusing the two
    with Reciprocal Rank Fusion so both semantic and lexical matches count.
    """

    def __init__(self, settings: QdrantSettings, embedder: Embedder):
        self.client = AsyncQdrantClient(host=settings.host, port=settings.port)
        self.settings = settings
        self.embedder = embedder
        self.sparse_text_embedding_ru: SparseTextEmbedding | None = None

    async def start(self, data: list[Document]):
        """Prepares the sparse index and populates Qdrant if it is empty.

        Args:
            data: Transcripts to index.
        """
        avg_doc_length = await asyncio.to_thread(self.get_avg_document_length, data)
        print(f"Average document length: {avg_doc_length}")
        self.sparse_text_embedding_ru = SparseTextEmbedding(
            "Qdrant/bm25", language="russian", avg_doc_length=avg_doc_length
        )

        if not await self.client.collection_exists(self.settings.collection_name):
            await self._create_collection(self.settings.collection_name)
        if (await self.client.count(self.settings.collection_name)).count == 0:
            await self._populate_collection(self.settings.collection_name, data)

    def _build_filter(
        self,
        *,
        year: int | None = None,
        meeting: str | None = None,
        topic: str | None = None,
        speaker: str | None = None,
    ) -> models.Filter | None:
        """Builds a conjunctive payload filter, skipping the fields left unset."""
        conditions = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in (
                ("year", year),
                ("meeting", meeting),
                ("topic", topic),
                ("speaker", speaker),
            )
            if value is not None
        ]
        return models.Filter(must=conditions) if conditions else None


    def get_avg_document_length(self, data: list[Document]) -> float:
        """Computes mean chunk length in tokens, needed for BM25 scoring.

        Args:
            data: Transcripts to measure.

        Returns:
            Average number of BM25 tokens per chunk.
        """
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
            """Embeds one batch of chunks and upserts it into Qdrant."""
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
        """Runs hybrid search and returns the best-matching chunks.

        Args:
            queries: Search queries; their results are fused together.
            top_k: Maximum chunks to return overall, capped at 10.
            year: Restrict to this year, if given.
            meeting: Restrict to this meeting, if given.
            topic: Restrict to this topic, if given.
            speaker: Restrict to this speaker, if given.

        Returns:
            JSON array of matching chunks with their metadata.
        """
        top_k = min(top_k, 10)
        flt = self._build_filter(
            year=year, meeting=meeting, topic=topic, speaker=speaker
        )

        async def embed_query(query: str):
            """Builds the dense and sparse vectors for one query."""
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

    async def discover(
        self,
        field: FacetField,
        *,
        year: int | None = None,
        meeting: str | None = None,
        topic: str | None = None,
        speaker: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        """Lists the distinct values of one indexed payload field, with hit counts.

        Used to ground filter arguments before searching: the caller can look up
        which years, meetings, topics or speakers actually exist (optionally
        within a narrower slice) instead of guessing a value that matches nothing.
        """
        if field not in FACET_FIELDS:
            raise ValueError(
                f"{field!r} is not facetable; expected one of {FACET_FIELDS}"
            )

        flt = self._build_filter(
            year=year, meeting=meeting, topic=topic, speaker=speaker
        )

        resp = await self.client.facet(
            collection_name=self.settings.collection_name,
            key=field,
            facet_filter=flt,
            limit=limit,
            exact=True,
        )

        return [{"value": hit.value, "count": hit.count} for hit in resp.hits]
