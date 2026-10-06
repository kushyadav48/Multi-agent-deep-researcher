"""Lazy shared corpus and bounded evidence for the existing research crew."""

from functools import lru_cache
from math import isfinite
from pathlib import Path

from rag.models import RetrievedChunk
from rag.service import RAGService
from rag.vector_store import ChromaVectorStore, validate_top_k


DEFAULT_TOP_K = 4
DEFAULT_MAX_DISTANCE = 0.6  # Chroma cosine distance: 1 - cosine similarity.
MAX_CONTEXT_CHARS = 6000
MAX_CHUNK_CHARS = 1000


@lru_cache(maxsize=1)
def get_default_rag_service() -> RAGService:
    """No database construction or model requests happen at import time."""
    return RAGService(vector_store=ChromaVectorStore(
        Path(__file__).resolve().parents[1] / "data" / "chroma",
    ))


def source_name(source: str) -> str:
    """Display filenames consistently for Windows and POSIX source paths."""
    return " ".join(source.replace("\\", "/").rsplit("/", 1)[-1].splitlines())


def document_citation(chunk: RetrievedChunk) -> str:
    page = f", p. {chunk.page}" if chunk.page is not None else ""
    return f"[Document: {source_name(chunk.source)}{page}]"


def format_document_context(chunks: list[RetrievedChunk]) -> str:
    """Render complete source headers, truncating only content to fit the cap."""
    context = "LOCAL DOCUMENT EVIDENCE"
    for index, chunk in enumerate(chunks, start=1):
        header = (
            f"\n\n[DOC {index}]\n"
            f"Source: {source_name(chunk.source)}\n"
            f"Page: {chunk.page if chunk.page is not None else 'N/A'}\n"
            f"Chunk: {chunk.chunk_index}\n"
            f"Citation: {document_citation(chunk)}\nContent:\n"
        )
        available = MAX_CONTEXT_CHARS - len(context) - len(header)
        if available <= 0:
            break
        content = chunk.text[:min(MAX_CHUNK_CHARS, available)]
        if content.strip():
            context += header + content
    return context if "\n\n[DOC " in context else ""


def retrieve_document_context(
    query: str, service: RAGService, *, top_k: int = DEFAULT_TOP_K,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> str:
    validate_top_k(top_k)
    if isinstance(max_distance, bool) or not isfinite(max_distance) or not 0 <= max_distance <= 2:
        raise ValueError("max_distance must be finite and between 0 and 2 (cosine distance)")
    if service.count() == 0:
        return ""
    chunks = service.retrieve(query, top_k=top_k)[:top_k]
    relevant = [chunk for chunk in chunks if (
        chunk.distance is not None and isfinite(chunk.distance)
        and chunk.distance <= max_distance
    )]
    return format_document_context(relevant)
