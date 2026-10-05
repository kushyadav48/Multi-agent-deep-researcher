import unittest
from unittest.mock import Mock, patch

import ollama

from rag.embeddings import EmbeddingError, OllamaEmbeddingProvider


class EmbeddingTests(unittest.TestCase):
    def make_provider(self, vectors):
        client = Mock()
        client.embed.return_value = ollama.EmbedResponse(embeddings=vectors)
        with patch("rag.embeddings.ollama.Client", return_value=client):
            provider = OllamaEmbeddingProvider()
        return provider, client

    def test_single_batch_and_empty_input(self):
        provider, client = self.make_provider([[1., 2.]])
        self.assertEqual(provider.embed_text("hello"), [1., 2.])
        client.embed.assert_called_once_with(model="qwen3-embedding:0.6b", input=["hello"], truncate=False)
        client.embed.return_value = ollama.EmbedResponse(embeddings=[[1., 2.], [3., 4.]])
        self.assertEqual(len(provider.embed_texts(["hello", "world"])), 2)
        self.assertEqual(provider.embed_texts([]), [])
        with self.assertRaises(ValueError):
            provider.embed_text(" ")

    def test_unavailable_and_failed_request(self):
        provider, client = self.make_provider([[1.]])
        for error, message in ((ConnectionError("offline"), "unavailable"),
                               (ollama.ResponseError("model missing", 404), "failed")):
            client.embed.side_effect = error
            with self.assertRaisesRegex(EmbeddingError, message):
                provider.embed_text("hello")

    def test_empty_malformed_and_changed_dimension(self):
        for vectors in ([], [[]], [[float("nan")]], [[1.], [1.]]):
            provider, _ = self.make_provider(vectors)
            with self.assertRaises(EmbeddingError):
                provider.embed_text("hello")
        provider, client = self.make_provider([[1., 2.]])
        provider.embed_text("first")
        client.embed.return_value = ollama.EmbedResponse(embeddings=[[1.]])
        with self.assertRaisesRegex(EmbeddingError, "dimension"):
            provider.embed_text("second")
