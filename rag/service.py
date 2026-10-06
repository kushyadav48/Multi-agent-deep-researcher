"""Document ingestion and similarity retrieval, without UI or agent coupling."""

from pathlib import Path
from dataclasses import replace

from rag.chunking import TextChunker
from rag.embeddings import EmbeddingProvider, OllamaEmbeddingProvider
from rag.loaders import DocumentLoadError, load_document
from rag.models import RetrievedChunk
from rag.vector_store import ChromaVectorStore, validate_top_k


class RAGService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: ChromaVectorStore | None = None,
        chunker: TextChunker | None = None,
    ):
        self.embedding_provider = embedding_provider if embedding_provider is not None else OllamaEmbeddingProvider()
        self.vector_store = vector_store if vector_store is not None else ChromaVectorStore()
        self.chunker = chunker if chunker is not None else TextChunker()

    def count(self) -> int:
        """Return stored chunks without contacting the embedding model."""
        return self.vector_store.count()

    def corpus_fingerprint(self) -> str:
        """Deterministic content identity without embedding or retrieval."""
        return self.vector_store.corpus_fingerprint()

    def ingest_file(self, path: str | Path, *, source: str | None = None) -> int:
        """Upsert chunks; optional stable source supports temporary uploads."""
        document = load_document(path)
        if source is not None:
            if not isinstance(source, str) or not source.strip():
                raise ValueError("source must be non-empty text")
            document = replace(document, source=source)
        chunks = self.chunker.chunk(document)
        if not chunks:
            raise DocumentLoadError(f"Document produced no non-empty chunks: {path}")
        # Bound local model request size even for longer PDFs.
        for start in range(0, len(chunks), 64):
            batch = chunks[start:start + 64]
            vectors = self.embedding_provider.embed_texts([chunk.text for chunk in batch])
            self.vector_store.upsert(batch, vectors)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Retrieval query must be non-empty text")
        validate_top_k(top_k)
        if self.count() == 0:
            return []
        return self.vector_store.search(self.embedding_provider.embed_text(query), top_k)
