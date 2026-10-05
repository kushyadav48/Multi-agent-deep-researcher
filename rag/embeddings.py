"""Reusable local embedding interface, also suitable for a future cache."""

from math import isfinite
from typing import Protocol

import ollama


class EmbeddingError(RuntimeError):
    """Ollama was unavailable or returned invalid embeddings."""


class EmbeddingProvider(Protocol):
    def embed_text(self, text: str) -> list[float]: ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


def validate_embeddings(vectors: list[list[float]], expected_count: int) -> int:
    """Reject count, shape, and numeric errors before storing or querying."""
    if len(vectors) != expected_count or not vectors:
        raise ValueError("Embedding count must match the non-empty input batch")
    dimension = len(vectors[0])
    if dimension == 0 or any(len(vector) != dimension for vector in vectors):
        raise ValueError("Embeddings must have a non-empty, consistent dimension")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
        for vector in vectors for value in vector
    ):
        raise ValueError("Embedding values must be finite numbers")
    return dimension


class OllamaEmbeddingProvider:
    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.model = model
        self._client = ollama.Client(host=base_url, timeout=timeout)
        self._dimension: int | None = None

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding input must contain non-empty text")
        try:
            response = self._client.embed(model=self.model, input=texts, truncate=False)
            vectors = response.embeddings
            dimension = validate_embeddings(vectors, len(texts))
            if self._dimension is not None and dimension != self._dimension:
                raise ValueError("Embedding dimension changed between requests")
            self._dimension = dimension
            return vectors
        except ConnectionError as error:
            raise EmbeddingError(
                "Ollama unavailable; start Ollama and check its local endpoint"
            ) from error
        except Exception as error:
            raise EmbeddingError(f"Embedding request failed for {self.model}: {error}") from error
