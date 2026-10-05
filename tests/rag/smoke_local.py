"""Opt-in live Ollama smoke test: python -m tests.rag.smoke_local.

Runs in a child process so Windows releases Chroma handles before temporary
files are removed. This module is not discovered by the offline test suite.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

import ollama


def run_smoke(root: Path) -> None:
    # Direct model validation precedes use of the project provider/store.
    client = ollama.Client(host="http://localhost:11434", timeout=120.0)
    started = perf_counter()
    first = client.embed(model="qwen3-embedding:0.6b", input="A local embedding smoke test.")
    direct_seconds = perf_counter() - started
    second = client.embed(model="qwen3-embedding:0.6b", input="A local embedding smoke test.")
    if not first.embeddings or not first.embeddings[0]:
        raise RuntimeError("Direct embedding request returned no vector")
    dimension = len(first.embeddings[0])
    if not second.embeddings or len(second.embeddings[0]) != dimension:
        raise RuntimeError("Repeated embedding request changed dimensionality")

    from rag.embeddings import OllamaEmbeddingProvider
    from rag.service import RAGService
    from rag.vector_store import ChromaVectorStore

    class TimedProvider:
        def __init__(self):
            self.provider = OllamaEmbeddingProvider()
            self.seconds: list[float] = []

        def embed_text(self, text):
            return self.embed_texts([text])[0]

        def embed_texts(self, texts):
            started = perf_counter()
            vectors = self.provider.embed_texts(texts)
            self.seconds.append(perf_counter() - started)
            return vectors

    provider = TimedProvider()
    store = ChromaVectorStore(root / "chroma")
    service = RAGService(provider, store)
    paths = []
    for name, text in (
        ("document_a.txt", "The Model Context Protocol allows applications to connect models with external tools and context."),
        ("document_b.txt", "Photosynthesis converts light energy into chemical energy in plants."),
    ):
        path = root / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
        service.ingest_file(path)
    service.ingest_file(paths[0])
    if store.count() != 2:
        raise RuntimeError("Repeated ingestion duplicated chunks")
    started = perf_counter()
    results = service.retrieve("What protocol connects models with tools and context?", top_k=2)
    retrieval_seconds = perf_counter() - started
    if len(results) != 2 or Path(results[0].source).name != "document_a.txt":
        raise RuntimeError("MCP document did not rank above photosynthesis")
    print(json.dumps({
        "model": "qwen3-embedding:0.6b",
        "embedding_dimension": dimension,
        "repeated_dimension_consistent": True,
        "direct_first_embedding_seconds": round(direct_seconds, 4),
        "document_embedding_seconds": [round(value, 4) for value in provider.seconds[:-1]],
        "query_embedding_seconds": round(provider.seconds[-1], 4),
        "retrieval_total_seconds": round(retrieval_seconds, 4),
        "vector_search_approx_seconds": round(retrieval_seconds - provider.seconds[-1], 4),
        "chunk_count_after_duplicate_ingestion": store.count(),
        "results": [{"source": Path(result.source).name, "text": result.text,
                     "page": result.page, "chunk_index": result.chunk_index,
                     "distance": result.distance} for result in results],
    }, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        run_smoke(Path(sys.argv[2]))
    else:
        with tempfile.TemporaryDirectory(prefix="deep_researcher_rag_smoke_") as directory:
            subprocess.run(
                [sys.executable, "-m", "tests.rag.smoke_local", "--worker", directory],
                check=True,
            )
