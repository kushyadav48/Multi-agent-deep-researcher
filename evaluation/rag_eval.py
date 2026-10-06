"""Production chunking, embeddings, cosine retrieval and relevance filtering."""

from time import perf_counter

from evaluation.datasets import ROOT
from evaluation.models import BenchmarkCaseResult
from rag.context import DEFAULT_MAX_DISTANCE, DEFAULT_TOP_K, retrieve_document_context
from rag.service import RAGService
from rag.vector_store import ChromaVectorStore


def prepare(provider, path):
    service = RAGService(provider, ChromaVectorStore(path))
    if service.count() == 0:
        for document in sorted((ROOT / "fixtures" / "rag").glob("*.md")):
            service.ingest_file(document, source=document.name)
    return service


def evaluate(case, service):
    retrieved = []
    started = perf_counter()
    retrieve_document_context(case["query"], service, top_k=DEFAULT_TOP_K,
                              max_distance=DEFAULT_MAX_DISTANCE,
                              observer=lambda status, chunks: retrieved.extend(chunks))
    milliseconds = (perf_counter() - started) * 1000
    rank = next((i for i, chunk in enumerate(retrieved, 1)
                 if chunk.source == case["expected_source"] and case["expected_fact"].casefold() in chunk.text.casefold()), None)
    return BenchmarkCaseResult(case["id"], "rag_retrieval", "PASS" if rank else "FAIL",
        duration_ms=milliseconds, measured=dict(rank=rank, retrieval_ms=milliseconds,
            expected_source=case["expected_source"], returned_chunks=len(retrieved),
            returned_sources=[chunk.source for chunk in retrieved],
            distances=[chunk.distance for chunk in retrieved]))
