from types import SimpleNamespace
from unittest.mock import Mock, patch

import agents
from semantic_cache.models import CacheScope
from semantic_cache.service import SemanticCacheService
from semantic_cache.store import ChromaCacheStore
from tests.rag.helpers import FakeEmbeddingProvider
from routing.models import FAST_SYNTHESIS_MODEL


def make_cache(tmp_path):
    return SemanticCacheService(FakeEmbeddingProvider(), ChromaCacheStore(tmp_path / 'cache'))


def test_hit_bypasses_all_downstream_work(tmp_path):
    cache = make_cache(tmp_path)
    rag = Mock()
    rag.corpus_fingerprint.return_value = 'corpus'
    scope = agents.research_cache_scope(rag, use_rag=True, top_k=4, max_distance=.6)
    cache.save('What is the protocol?', 'cached answer', scope)
    with patch('agents.Crew.kickoff') as kickoff, patch('agents.DDGS') as ddgs, \
         patch('agents.get_llm_client') as llm, \
         patch('agents.create_research_crew') as factory:
        assert agents.run_research('Explain the protocol.', rag_service=rag, cache_service=cache) == 'cached answer'
        for spy in (kickoff, ddgs, llm, factory, rag.retrieve, rag.count):
            spy.assert_not_called()


def test_disabled_does_zero_cache_work(tmp_path):
    cache = Mock()
    with patch('agents.get_default_cache_service') as default, \
         patch('agents.create_research_crew', return_value=Mock(kickoff=Mock(return_value=SimpleNamespace(raw='answer')))):
        assert agents.run_research('query', use_rag=False, use_cache=False, cache_service=cache) == 'answer'
    default.assert_not_called()
    assert not cache.mock_calls


def test_miss_runs_pipeline_and_stores(tmp_path):
    cache = make_cache(tmp_path)
    crew = Mock(kickoff=Mock(return_value=SimpleNamespace(raw='answer')))
    rag = Mock()
    rag.corpus_fingerprint.return_value = 'corpus'
    rag.count.return_value = 0
    with patch('agents.create_research_crew', return_value=crew) as factory:
        assert agents.run_research('protocol', rag_service=rag, cache_service=cache) == 'answer'
    crew.kickoff.assert_called_once()
    factory.assert_called_once_with('protocol', document_context='', synthesis_model=FAST_SYNTHESIS_MODEL)
    rag.count.assert_called_once()
    assert cache.store.count() == 1


def test_failed_empty_and_fallback_results_not_cached(tmp_path):
    cache = make_cache(tmp_path)
    for result in ('', 'Error: failed', 'Error occurred while searching: failed'):
        with patch('agents.create_research_crew', return_value=Mock(kickoff=Mock(return_value=SimpleNamespace(raw=result)))):
            agents.run_research('protocol', use_rag=False, cache_service=cache)
    with patch('agents.create_research_crew', return_value=Mock(kickoff=Mock(
        return_value=SimpleNamespace(raw='incomplete', tasks_output=[SimpleNamespace(raw='one task')])
    ))):
        agents.run_research('protocol', use_rag=False, cache_service=cache)
    with patch('agents.create_research_crew', side_effect=RuntimeError('failure')):
        assert agents.run_research('protocol', use_rag=False, cache_service=cache).startswith('Error:')
    rag = Mock()
    rag.corpus_fingerprint.return_value = 'corpus'
    rag.count.side_effect = RuntimeError('retrieval failed')
    with patch('agents.create_research_crew', return_value=Mock(kickoff=Mock(return_value=SimpleNamespace(raw='answer')))):
        assert 'web-only research' in agents.run_research('protocol', rag_service=rag, cache_service=cache)
    assert cache.store.count() == 0


def test_cache_and_fingerprint_failure_preserve_research(tmp_path):
    cache = Mock()
    cache.lookup.side_effect = RuntimeError('cache offline')
    rag = Mock()
    rag.corpus_fingerprint.side_effect = RuntimeError('fingerprint unavailable')
    rag.count.return_value = 0
    with patch('agents.create_research_crew', return_value=Mock(kickoff=Mock(return_value=SimpleNamespace(raw='answer')))):
        assert agents.run_research('query', use_rag=False, cache_service=cache) == 'answer'
        cache.save.assert_not_called()
        assert agents.run_research('query', rag_service=rag, cache_service=cache) == 'answer'
        cache.save.assert_not_called()


def test_corpus_changes_during_research_not_cached(tmp_path):
    cache = make_cache(tmp_path)
    rag = Mock()
    rag.corpus_fingerprint.side_effect = ['before', 'after']
    rag.count.return_value = 0
    with patch('agents.create_research_crew', return_value=Mock(kickoff=Mock(return_value=SimpleNamespace(raw='answer')))):
        assert agents.run_research('protocol', rag_service=rag, cache_service=cache) == 'answer'
    assert cache.store.count() == 0


def test_write_failure_preserves_successful_answer(tmp_path):
    cache = make_cache(tmp_path)
    with patch.object(cache, 'save', side_effect=RuntimeError('disk full')), \
         patch('agents.create_research_crew', return_value=Mock(kickoff=Mock(return_value=SimpleNamespace(raw='answer')))):
        assert agents.run_research('protocol', use_rag=False, cache_service=cache) == 'answer'
