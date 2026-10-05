import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.models import DocumentChunk
from rag.vector_store import ChromaVectorStore, VectorStoreError
from tests.rag.helpers import FakeEmbeddingProvider


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        # Chroma may retain Windows file handles until process exit.
        self.temp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "chroma"
        self.store = ChromaVectorStore(self.path)
        self.provider = FakeEmbeddingProvider()
        self.chunks = [
            DocumentChunk("a", "Models connect to tools using a protocol.", "paper.pdf", 2, 0,
                          {"title": "MCP", "source": "custom", "page": 99, "reviewed": True}),
            DocumentChunk("b", "Plants use photosynthesis.", "plants.txt", None, 0),
        ]
        self.vectors = self.provider.embed_texts([chunk.text for chunk in self.chunks])

    def test_ranking_metadata_and_repeated_upsert(self):
        self.store.upsert(self.chunks, self.vectors)
        self.store.upsert(self.chunks, self.vectors)
        self.assertEqual(self.store.count(), 2)
        results = self.store.search(self.provider.embed_text("protocol tools"), top_k=5)
        self.assertEqual([result.source for result in results], ["paper.pdf", "plants.txt"])
        self.assertEqual(results[0].page, 2)
        self.assertEqual(results[0].chunk_index, 0)
        self.assertEqual(dict(results[0].metadata), dict(self.chunks[0].metadata))
        self.assertIsNone(results[1].page)
        self.assertLess(results[0].distance, results[1].distance)

    def test_persistence_in_new_process(self):
        import json
        import subprocess
        import sys
        self.store.upsert(self.chunks, self.vectors)
        script = (
            "import json,sys; from rag.vector_store import ChromaVectorStore; "
            "s=ChromaVectorStore(sys.argv[1]); "
            "print(json.dumps([s.count(),s.search([1.,0.,.1],1)[0].source]))"
        )
        output = subprocess.check_output([sys.executable, "-c", script, str(self.path)], text=True)
        self.assertEqual(json.loads(output), [2, "paper.pdf"])

    def test_empty_top_k_validation_and_clear(self):
        self.assertEqual(self.store.search([1., 0., .1]), [])
        self.store.upsert(self.chunks, self.vectors)
        self.assertEqual(len(self.store.search([1., 0., .1], top_k=1)), 1)
        for top_k in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                self.store.search([1., 0., .1], top_k)
        self.store.clear()
        self.assertEqual(self.store.count(), 0)

    def test_bad_embeddings_and_backend_failure_are_explicit(self):
        for vectors in ([], [[], []], [[1., 0.], [1.]], [[float("nan")], [1.]]):
            with self.assertRaises(ValueError):
                self.store.upsert(self.chunks, vectors)
        with patch.object(self.store._collection, "upsert", side_effect=RuntimeError("disk failure")):
            with self.assertRaisesRegex(VectorStoreError, "upsert"):
                self.store.upsert(self.chunks, self.vectors)
