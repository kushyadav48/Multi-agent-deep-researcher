from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from semantic_cache.models import CacheScope
from semantic_cache.service import SemanticCacheService
from semantic_cache.store import ChromaCacheStore


QUERY = 'What is the Model Context Protocol?'
PARAPHRASE = 'Explain the Model Context Protocol.'


@pytest.fixture
def cache(tmp_path):
    provider = Mock()
    provider.embed_text.side_effect = lambda text: (
        [0., 1.] if 'photosynthesis' in text else [1., 0.]
    )
    return SemanticCacheService(provider, ChromaCacheStore(tmp_path / 'cache'), clock=lambda: 1000.)


def test_empty_exact_and_paraphrase(cache):
    scope = CacheScope(use_rag=False)
    with patch.object(cache.store, 'search', wraps=cache.store.search) as search:
        assert cache.lookup(QUERY, scope).hit is None
        search.assert_not_called()
    cache.embedding_provider.embed_text.assert_not_called()
    assert cache.save(QUERY, 'answer', scope)
    cache.embedding_provider.embed_text.reset_mock()
    hit = cache.lookup('  What is the Model   Context Protocol?  ', scope).hit
    assert hit.kind == 'exact' and hit.entry.answer == 'answer'
    cache.embedding_provider.embed_text.assert_not_called()
    hit = cache.lookup(PARAPHRASE, scope).hit
    assert hit.kind == 'semantic' and hit.similarity > .97
    assert hit.entry.query == QUERY and len(hit.entry.embedding) == 2


def test_unrelated_threshold_and_hard_negatives(cache):
    scope = CacheScope(use_rag=False)
    cache.save(QUERY, 'answer', scope)
    assert cache.lookup('How does photosynthesis work?', scope).hit is None
    cache.embedding_provider.embed_text.return_value = [0.96, 0.28]
    cache.embedding_provider.embed_text.side_effect = None
    assert cache.lookup(PARAPHRASE, scope).hit is None
    cache.embedding_provider.embed_text.side_effect = lambda text: [1., 0.]
    for positive, negative in (
        ('What are the advantages of MCP?', 'What are the disadvantages of MCP?'),
        ('How do I enable caching?', 'How do I disable caching?'),
        ('Explain TCP connection establishment.', 'Explain TCP connection termination.'),
        ('Explain Python 3.11.', 'Explain Python 3.12.'),
    ):
        cache.save(positive, 'answer', scope)
        assert cache.lookup(negative, scope).hit is None


def test_ttl_and_current_ttl_override(cache):
    scope = CacheScope(use_rag=False)
    cache.save(QUERY, 'answer', scope)
    cache.clock = lambda: 4600.
    assert cache.lookup(QUERY, scope).hit is None
    assert cache.lookup(PARAPHRASE, scope).hit is None
    cache.clock = lambda: 1000.
    cache.save(QUERY, 'answer', scope)
    cache.ttl_seconds = 10
    cache.clock = lambda: 1010.
    assert cache.lookup(QUERY, scope).hit is None


def test_scope_invalidation(cache):
    scope = CacheScope(use_rag=True, corpus_fingerprint='corpus-a')
    cache.save(QUERY, 'answer', scope)
    for other in (
        replace(scope, research_model='other'), replace(scope, version='1'),
        replace(scope, use_rag=False), replace(scope, corpus_fingerprint='corpus-b'),
        replace(scope, rag_top_k=1), replace(scope, rag_max_distance=.2),
        replace(scope, embedding_model='other-embedding'),
        replace(scope, rag_embedding_model='other-rag-embedding'),
    ):
        assert cache.lookup(QUERY, other).hit is None
    assert CacheScope(use_rag=False, corpus_fingerprint='ignored').digest == CacheScope(use_rag=False).digest
    assert scope.digest == replace(scope).digest


def test_save_filters_errors_duplicates_and_reuses_miss_vector(cache):
    scope = CacheScope(use_rag=False)
    for answer in ('', '  ', 'Error: failed', 'Error occurred while searching: failed', None):
        assert not cache.save(QUERY, answer, scope)
    assert cache.store.count() == 0
    cache.save(QUERY, 'first', scope)
    miss = cache.lookup('Describe MCP.', scope)
    # Make a new topic vector that misses the existing answer.
    cache.embedding_provider.embed_text.side_effect = lambda text: [0., 1.]
    miss = cache.lookup('New topic', scope)
    cache.embedding_provider.embed_text.reset_mock()
    cache.save('New topic', 'new answer', scope, embedding=miss.embedding)
    cache.embedding_provider.embed_text.assert_not_called()
    cache.save('  What is the Model Context Protocol? ', 'replacement', scope)
    assert cache.store.count() == 2
    assert cache.lookup(QUERY, scope).hit.entry.answer == 'replacement'


@pytest.mark.parametrize('kwargs', [dict(ttl_seconds=0), dict(ttl_seconds=float('nan')),
                                     dict(similarity_threshold=2), dict(similarity_threshold=True)])
def test_invalid_configuration(cache, kwargs):
    with pytest.raises(ValueError):
        SemanticCacheService(cache.embedding_provider, cache.store, **kwargs)


def test_threshold_is_strict_and_expired_nearest_is_filtered(cache):
    scope = CacheScope(use_rag=False)
    cache.save(QUERY, 'older', scope)
    cache.clock = lambda: 1010.
    cache.save('Describe the Model Context Protocol.', 'newer', scope)
    cache.clock = lambda: 4600.
    hit = cache.lookup(PARAPHRASE, scope).hit
    assert hit.entry.answer == 'newer'
    with patch.object(cache.store, 'search', return_value=(hit.entry, cache.similarity_threshold)):
        assert cache.lookup(PARAPHRASE, scope).hit is None
    cache.similarity_threshold = .9
    with patch.object(cache.store, 'search', return_value=(hit.entry, .95)):
        assert cache.lookup(PARAPHRASE, scope).hit is not None
