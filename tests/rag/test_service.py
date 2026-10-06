import tempfile
import unittest
from unittest.mock import Mock
from pathlib import Path

from rag.chunking import TextChunker
from rag.service import RAGService
from rag.vector_store import ChromaVectorStore
from tests.rag.helpers import FakeEmbeddingProvider, write_pdf


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ChromaVectorStore(self.root / "chroma")
        self.service = RAGService(FakeEmbeddingProvider(), self.store, TextChunker())

    def test_ingest_retrieve_duplicates_and_top_k(self):
        for name, text in (("mcp.md", "Models connect to tools using a protocol."),
                           ("plants.txt", "Plants use photosynthesis.")):
            path = self.root / name
            path.write_text(text, encoding="utf-8")
            self.assertEqual(self.service.ingest_file(path), 1)
            self.assertEqual(self.service.ingest_file(path), 1)
        self.assertEqual(self.store.count(), 2)
        results = self.service.retrieve("What protocol connects tools?", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, str((self.root / "mcp.md").resolve()))
        self.assertIn("protocol", results[0].text)

    def test_pdf_ingest_returns_page_metadata(self):
        path = self.root / "paper.pdf"
        write_pdf(path)
        self.assertEqual(self.service.ingest_file(path), 2)
        result = self.service.retrieve("photosynthesis plants", 1)[0]
        self.assertEqual(result.page, 2)
        self.assertEqual(result.chunk_index, 1)

    def test_invalid_query_and_top_k(self):
        for query in ("", " \n"):
            with self.assertRaises(ValueError):
                self.service.retrieve(query)
        with self.assertRaises(ValueError):
            self.service.retrieve("protocol", 0)

    def test_empty_corpus_does_not_embed(self):
        provider = Mock()
        service = RAGService(provider, self.store)
        self.assertEqual(service.count(), 0)
        self.assertEqual(service.retrieve("protocol"), [])
        provider.embed_text.assert_not_called()
        provider.embed_texts.assert_not_called()

    def test_temporary_upload_sources_remain_stable(self):
        for directory in ("upload_one", "upload_two"):
            path = self.root / directory / "notes.md"
            path.parent.mkdir()
            path.write_text("Models connect to tools using a protocol.", encoding="utf-8")
            self.service.ingest_file(path, source="notes.md")
        self.assertEqual(self.service.count(), 1)
        self.assertEqual(self.service.retrieve("protocol", 1)[0].source, "notes.md")
