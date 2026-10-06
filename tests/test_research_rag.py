import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from crewai import Process

import agents
from rag.context import MAX_CONTEXT_CHARS, document_citation, format_document_context
from rag.models import RetrievedChunk
from rag.embeddings import EmbeddingError
from rag.vector_store import VectorStoreError
from semantic_cache.models import CacheLookup


def chunk(source="protocol_notes.pdf", page=3, distance=0.2, text="Cedar-47 is the protocol codename."):
    return RetrievedChunk(text, source, page, 7, distance)


class ResearchRAGTests(unittest.TestCase):
    def setUp(self):
        self.service = Mock()
        self.service.corpus_fingerprint.return_value = "test-corpus"
        cache = patch("agents.get_default_cache_service", return_value=Mock(lookup=Mock(return_value=CacheLookup())))
        cache.start()
        self.addCleanup(cache.stop)
        self.service.count.return_value = 1
        self.service.retrieve.return_value = [chunk()]
        self.kickoff = patch("agents.Crew.kickoff", return_value=SimpleNamespace(raw="grounded answer"))
        self.kickoff.start()
        self.addCleanup(self.kickoff.stop)
        self.crews = []
        real_factory = agents.create_research_crew

        def capture(*args, **kwargs):
            crew = real_factory(*args, **kwargs)
            self.crews.append(crew)
            return crew

        factory = patch("agents.create_research_crew", side_effect=capture)
        factory.start()
        self.addCleanup(factory.stop)

    def run_query(self, **kwargs):
        return agents.run_research("What is the codename?", rag_service=self.service, **kwargs)

    def test_empty_corpus_runs_web_without_retrieval(self):
        self.service.count.return_value = 0
        self.assertEqual(self.run_query(), "grounded answer")
        self.service.retrieve.assert_not_called()
        self.assertNotIn("LOCAL DOCUMENT", self.crews[0].tasks[1].description)
        self.assertTrue(self.crews[0].tasks[0].tools)

    def test_disabled_does_not_construct_or_call_rag(self):
        with patch("agents.get_default_rag_service") as default:
            self.assertEqual(self.run_query(use_rag=False), "grounded answer")
            default.assert_not_called()
        self.service.count.assert_not_called()
        self.service.retrieve.assert_not_called()

    def test_backward_compatible_call_uses_default_service(self):
        with patch("agents.get_default_rag_service", return_value=self.service) as default:
            self.assertEqual(agents.run_research("codename"), "grounded answer")
        default.assert_called_once_with()
        self.service.retrieve.assert_called_once_with("codename", top_k=4)

    def test_relevant_evidence_reaches_analyst_and_writer_instructions(self):
        self.run_query()
        search, analyst, writer = self.crews[0].tasks
        self.assertNotIn("Cedar-47", search.description)
        self.assertIn("Cedar-47", analyst.description)
        self.assertIn("Source: protocol_notes.pdf\nPage: 3\nChunk: 7", analyst.description)
        self.assertIn("[Document: protocol_notes.pdf, p. 3]", analyst.description)
        self.assertIn("preserving", writer.description)
        self.assertIn("[Document: filename]", writer.description)
        self.assertIn("Cedar-47", writer.description)
        self.assertIn("[Document: protocol_notes.pdf, p. 3]", writer.description)
        self.assertIn("conflict", analyst.description)

    def test_nonpaged_citation(self):
        self.service.retrieve.return_value = [chunk("notes.md", None)]
        self.run_query()
        description = self.crews[0].tasks[1].description
        self.assertIn("[Document: notes.md]", description)
        self.assertIn("Page: N/A", description)
        self.assertNotIn("p. None", description)
        self.assertIn("[Document: notes.md]", self.crews[0].tasks[2].description)

    def test_irrelevant_and_unknown_distances_excluded(self):
        self.service.retrieve.return_value = [chunk(distance=d) for d in (0.84, None, float("nan"))]
        self.run_query()
        self.assertNotIn("LOCAL DOCUMENT", self.crews[0].tasks[1].description)
        self.assertIn("No local document evidence was retrieved", self.crews[0].tasks[2].description)

    def test_top_k_threshold_and_context_cap(self):
        self.service.retrieve.return_value = [chunk(f"{i}.md", None, 0.5, "x" * 2000) for i in range(8)]
        self.run_query(rag_top_k=2, rag_max_distance=0.5)
        self.service.retrieve.assert_called_once_with("What is the codename?", top_k=2)
        description = self.crews[0].tasks[1].description
        self.assertIn("[DOC 2]", description)
        self.assertNotIn("[DOC 3]", description)
        context = format_document_context(self.service.retrieve.return_value)
        self.assertLessEqual(len(context), MAX_CONTEXT_CHARS)
        self.assertNotIn("x" * 1001, context)
        self.assertEqual(context, format_document_context(self.service.retrieve.return_value))

    def test_embedding_and_store_failures_explicit_web_fallback(self):
        for error in (EmbeddingError("offline"), VectorStoreError("broken store")):
            with self.subTest(error=error):
                self.service.retrieve.side_effect = error
                with self.assertLogs("agents", level="WARNING") as logs:
                    result = self.run_query()
                self.assertIn("web-only research", result)
                self.assertIn(str(error), logs.output[0])
                self.assertNotIn("LOCAL DOCUMENT", self.crews[-1].tasks[1].description)

    def test_default_construction_failure_keeps_web_available(self):
        with patch("agents.get_default_rag_service", side_effect=VectorStoreError("cannot open")):
            with self.assertLogs("agents", level="WARNING"):
                result = agents.run_research("codename")
        self.assertIn("web-only research", result)

    def test_three_agents_sequential_task_context_preserved(self):
        self.run_query()
        crew = self.crews[0]
        self.assertEqual([a.role for a in crew.agents], ["Web Searcher", "Research Analyst", "Technical Writer"])
        self.assertEqual(crew.process, Process.sequential)
        self.assertEqual([t.agent for t in crew.tasks], crew.agents)
        self.assertEqual(crew.tasks[1].context, [crew.tasks[0]])
        self.assertEqual(crew.tasks[2].context, crew.tasks[:2])
        self.assertEqual(crew.agents[0].llm.model, "qwen2.5:3b")  # CrewAI normalizes the provider prefix.

    def test_document_citation_deterministic(self):
        self.assertEqual(document_citation(chunk(r"C:\docs\protocol_notes.pdf")), "[Document: protocol_notes.pdf, p. 3]")
        self.assertEqual(document_citation(chunk("/docs/notes.md", None)), "[Document: notes.md]")


if __name__ == "__main__":
    unittest.main()
