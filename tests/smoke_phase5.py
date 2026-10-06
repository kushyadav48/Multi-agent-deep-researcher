"""Opt-in live regression: python -m tests.smoke_phase5 (not unit discovery)."""

from contextlib import redirect_stdout
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch


def run_smoke(root: Path):
    import ollama
    import agents
    from ddgs.ddgs import DDGS as ConcreteDDGS
    from crewai import Process
    from crewai.events import crewai_event_bus
    from rag.context import retrieve_document_context
    from rag.embeddings import OllamaEmbeddingProvider
    from rag.service import RAGService
    from rag.vector_store import ChromaVectorStore

    query = "What is Project Orion's internal protocol codename?"
    fact = "Cedar-47"
    filename = "orion_protocol.md"
    path = root / filename
    path.write_text(
        "Internal protocol notes. Project Orion's internal protocol codename is Cedar-47.",
        encoding="utf-8",
    )
    started = perf_counter()
    first = ollama.Client(timeout=120).embed(model="qwen3-embedding:0.6b", input=query)
    first_seconds = perf_counter() - started

    class TimedProvider(OllamaEmbeddingProvider):
        def __init__(self):
            super().__init__()
            self.times = []

        def embed_texts(self, texts):
            started = perf_counter()
            vectors = super().embed_texts(texts)
            self.times.append(perf_counter() - started)
            return vectors

    class TimedStore(ChromaVectorStore):
        def __init__(self):
            super().__init__(root / "chroma")
            self.times = []
            self.results = []

        def search(self, embedding, top_k=5):
            started = perf_counter()
            results = super().search(embedding, top_k)
            self.times.append(perf_counter() - started)
            self.results = results
            return results

    provider, store = TimedProvider(), TimedStore()
    service = RAGService(provider, store)
    service.ingest_file(path)
    started = perf_counter()
    context = retrieve_document_context(query, service)
    context_total_seconds = perf_counter() - started
    if fact not in context:
        raise AssertionError("Unique fact failed the relevance filter")
    context_format_seconds = context_total_seconds - provider.times[-1] - store.times[-1]

    searches, crews = [], []
    # ddgs.DDGS is a lazy proxy; observe the class actually instantiated.
    real_text, real_create = ConcreteDDGS.text, agents.create_research_crew

    def observe_search(instance, *args, **kwargs):
        try:
            results = real_text(instance, *args, **kwargs)
        except Exception as error:
            searches.append({"query": str(args[0] if args else kwargs), "error": str(error)})
            raise
        searches.append({"query": str(args[0] if args else kwargs), "results": len(results)})
        return results

    def observe_crew(*args, **kwargs):
        crew = real_create(*args, **kwargs)
        crews.append(crew)
        return crew

    started = perf_counter()
    with patch.object(ConcreteDDGS, "text", observe_search), patch.object(agents, "create_research_crew", observe_crew):
        with redirect_stdout(sys.stderr):
            answer = agents.run_research(query, rag_service=service)
            if not crewai_event_bus.flush(timeout=30):
                raise AssertionError("CrewAI event bus did not flush")
    research_seconds = perf_counter() - started
    citation = f"[Document: {filename}]"
    crew = crews[0]
    checks = {
        "local_document_retrieved": any(
            Path(result.source).name == filename and fact in result.text
            for result in store.results
        ),
        "local_evidence_reaches_analyst": (
            fact in crew.tasks[1].description and citation in crew.tasks[1].description
        ),
        "local_evidence_reaches_writer": (
            fact in crew.tasks[2].description and citation in crew.tasks[2].description
        ),
        "unique_fact_in_answer": fact in answer,
        "correct_document_citation": citation in answer,
        "no_fabricated_page_number": (
            all(result.page is None for result in store.results)
            and all(value == citation for value in re.findall(r"\[Document: [^\]]+\]", answer))
        ),
        "ddgs_executed": bool(searches),
        "ddgs_successful_call": any("results" in search for search in searches),
        "three_agents_sequential": len(crew.agents) == 3 and crew.process == Process.sequential,
        "all_tasks_completed": all(task.output is not None for task in crew.tasks),
        "analyst_completed": crew.tasks[1].output is not None,
        "writer_completed": crew.tasks[2].output is not None,
        "analyst_used_fact": fact in crew.tasks[1].output.raw if crew.tasks[1].output else False,
        "analyst_preserved_citation": citation in crew.tasks[1].output.raw if crew.tasks[1].output else False,
        "no_fallback_or_error": not answer.startswith("Error:") and "knowledge base unavailable" not in answer,
    }
    report = {
        "query": query, "answer": answer, "checks": checks,
        "source": Path(store.results[0].source).name,
        "distance": store.results[0].distance,
        "embedding_dimension": len(first.embeddings[0]),
        "initial_embedding_request_seconds": first_seconds,
        "initial_model_load_seconds": first.load_duration / 1e9,
        "cold_load_observation": "Initial Ollama-reported load duration; model may already be resident.",
        "warm_query_embedding_seconds": provider.times[-1],
        "vector_retrieval_seconds": store.times[-1],
        "context_preparation_seconds": context_format_seconds,
        "context_including_embedding_retrieval_seconds": context_total_seconds,
        "total_research_seconds": research_seconds,
        "ddgs_calls": searches,
    }
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise AssertionError("Real Phase 5 regression failed; see checks above")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        run_smoke(Path(sys.argv[2]))
    else:
        # The child exit releases Windows Chroma handles before cleanup.
        with tempfile.TemporaryDirectory(prefix="deep_researcher_phase5_") as directory:
            subprocess.run([sys.executable, "-m", "tests.smoke_phase5", "--worker", directory], check=True)
