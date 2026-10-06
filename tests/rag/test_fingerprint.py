from dataclasses import replace
from unittest.mock import Mock

from rag.models import DocumentChunk
from rag.service import RAGService
from rag.vector_store import ChromaVectorStore
from semantic_cache.models import CacheScope
from semantic_cache.service import SemanticCacheService
from semantic_cache.store import ChromaCacheStore
from tests.rag.helpers import FakeEmbeddingProvider


def test_empty_deterministic_and_content_metadata_invalidation(tmp_path):
    store = ChromaVectorStore(tmp_path / 'corpus')
    other = ChromaVectorStore(tmp_path / 'other')
    empty = store.corpus_fingerprint()
    assert empty == other.corpus_fingerprint()
    provider = FakeEmbeddingProvider()
    service = RAGService(provider, store)
    cache = SemanticCacheService(provider, ChromaCacheStore(tmp_path / 'cache'))
    chunk = DocumentChunk('same-id', 'protocol text', 'notes.md', None, 0)
    store.upsert([chunk], [[1., 0., .1]])
    first = service.corpus_fingerprint()
    scope = CacheScope(corpus_fingerprint=first)
    cache.save('protocol', 'answer', scope)
    assert cache.lookup('protocol', scope).hit is not None
    store.upsert([replace(chunk, text='Changed protocol content')], [[1., 0., .1]])
    second = service.corpus_fingerprint()
    assert store.count() == 1 and first != second != empty
    assert cache.lookup('protocol', CacheScope(corpus_fingerprint=second)).hit is None
    store.upsert([replace(chunk, source='renamed.md', page=2)], [[1., 0., .1]])
    assert service.corpus_fingerprint() not in (first, second)
    # Neither insertion order nor a new service instance affects the digest.
    store.clear()
    chunks = [chunk, replace(chunk, id='other-id')]
    store.upsert(chunks, [[1., 0., .1]] * 2)
    other.upsert(list(reversed(chunks)), [[1., 0., .1]] * 2)
    assert store.corpus_fingerprint() == other.corpus_fingerprint()


def test_service_fingerprint_does_not_embed_or_retrieve():
    provider, store = Mock(), Mock()
    store.corpus_fingerprint.return_value = 'digest'
    assert RAGService(provider, store).corpus_fingerprint() == 'digest'
    assert not provider.mock_calls
    store.search.assert_not_called()
