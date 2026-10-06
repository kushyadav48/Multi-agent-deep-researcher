"""Embedded persistent Chroma; all vectors come from the supplied provider."""

import json
from hashlib import sha256
from pathlib import Path

import chromadb
from chromadb.config import Settings

from rag.embeddings import validate_embeddings
from rag.models import DocumentChunk, RetrievedChunk


class VectorStoreError(RuntimeError):
    """A persistent vector-store operation failed."""


def validate_top_k(top_k: int) -> None:
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


class ChromaVectorStore:
    def __init__(
        self,
        path: str | Path = "data/chroma",
        collection_name: str = "deep_researcher_documents",
    ):
        self.path = Path(path).expanduser().resolve()
        self.collection_name = collection_name
        try:
            self._client = chromadb.PersistentClient(
                path=str(self.path), settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._get_collection()
        except Exception as error:
            raise VectorStoreError(f"Unable to open vector store at {self.path}: {error}") from error

    def _get_collection(self):
        return self._client.get_or_create_collection(
            name=self.collection_name, embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception as error:
            raise VectorStoreError(f"Vector-store count failed: {error}") from error

    def corpus_fingerprint(self) -> str:
        """Hash current text and citation metadata, including same-ID edits."""
        try:
            result = self._collection.get(include=["documents", "metadatas"])
            rows = sorted(zip(result["ids"], result["documents"], result["metadatas"], strict=True))
            payload = json.dumps(rows, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":"), allow_nan=False)
            return sha256(payload.encode("utf-8")).hexdigest()
        except Exception as error:
            raise VectorStoreError(f"Corpus fingerprint failed: {error}") from error

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if not chunks and not embeddings:
            return
        validate_embeddings(embeddings, len(chunks))
        if any(not chunk.id or not chunk.text.strip() for chunk in chunks):
            raise ValueError("Chunks must have non-empty IDs and text")
        # Keep caller metadata in its own payload so keys such as page/source
        # cannot override canonical fields, and round-trip exactly.
        metadatas = []
        for chunk in chunks:
            metadata = {
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "metadata_json": json.dumps(dict(chunk.metadata), ensure_ascii=False),
            }
            if chunk.page is not None:
                metadata["page"] = chunk.page
            metadatas.append(metadata)
        try:
            batch_size = self._client.get_max_batch_size()
            for start in range(0, len(chunks), batch_size):
                end = start + batch_size
                self._collection.upsert(
                    ids=[chunk.id for chunk in chunks[start:end]],
                    documents=[chunk.text for chunk in chunks[start:end]],
                    embeddings=embeddings[start:end], metadatas=metadatas[start:end],
                )
        except Exception as error:
            raise VectorStoreError(f"Vector-store upsert failed: {error}") from error

    def search(self, embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        validate_top_k(top_k)
        validate_embeddings([embedding], 1)
        count = self.count()
        if count == 0:
            return []
        try:
            response = self._collection.query(
                query_embeddings=[embedding], n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
            return [
                RetrievedChunk(
                    text=text, source=metadata["source"], page=metadata.get("page"),
                    chunk_index=metadata["chunk_index"], distance=distance,
                    metadata=json.loads(metadata["metadata_json"]),
                )
                for text, metadata, distance in zip(
                    response["documents"][0], response["metadatas"][0],
                    response["distances"][0], strict=True,
                )
            ]
        except Exception as error:
            raise VectorStoreError(f"Vector-store similarity search failed: {error}") from error

    def clear(self) -> None:
        """Delete only this collection and recreate it for development/tests."""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._get_collection()
        except Exception as error:
            raise VectorStoreError(f"Vector-store clear failed: {error}") from error
