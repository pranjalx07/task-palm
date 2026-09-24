from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from qdrant_client import QdrantClient, models

from app.config import Settings


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self.collection = settings.qdrant_collection

    def upsert_chunks(
        self, document_id: int, chunks: Sequence[str], embeddings: Sequence[Sequence[float]]
    ) -> None:
        if not chunks:
            return
        vector_size = len(embeddings[0])
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
        points = [
            models.PointStruct(
                id=str(uuid4()),
                vector=list(embedding),
                payload={"document_id": document_id, "chunk_index": index, "text": text},
            )
            for index, (text, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, embedding: Sequence[float], limit: int = 5) -> list[dict[str, str]]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=list(embedding),
            limit=limit,
            with_payload=True,
        )
        return [
            {"text": str(point.payload.get("text", "")), "score": str(point.score)}
            for point in response.points
            if point.payload
        ]


def build_vector_store(settings: Settings) -> VectorStore:
    return VectorStore(settings)