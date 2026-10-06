from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock, patch
import json

import pytest
from crewai import Process
from ddgs.ddgs import DDGS as ConcreteDDGS

import agents
from routing import ModelRoute, route_query
from routing.models import FAST_SYNTHESIS_MODEL, QUALITY_SYNTHESIS_MODEL, SEARCH_MODEL
from semantic_cache.models import CacheScope, digest, normalize_query
from semantic_cache.service import SemanticCacheService
from semantic_cache.store import ChromaCacheStore
from tests.rag.helpers import FakeEmbeddingProvider
from tests.routing.test_router import SIMPLE, COMPLEX


def scope(query=SIMPLE, mode='auto'):
    return agents.research_cache_scope(
        None, use_rag=False, top_k=4, max_distance=.6,
        routing_decision=route_query(query, mode),
    )


@pytest.mark.parametrize('query,route,model', [
    (SIMPLE, ModelRoute.FAST, FAST_SYNTHESIS_MODEL),
    (COMPLEX, ModelRoute.QUALITY, QUALITY_SYNTHESIS_MODEL),
])
def test_constructed_agent_models_and_sequential_evidence(query, route, model):
    decision = route_query(query)
    with patch('crewai.crew.Crew.kickoff') as kickoff:
        crew = agents.create_research_crew(
            query, document_context='[Document: notes.md] Evidence.',
            synthesis_model=decision.synthesis_model,
        )
    kickoff.assert_not_called()
    assert decision.selected_route is route
    assert [a.role for a in crew.agents] == ['Web Searcher', 'Research Analyst', 'Technical Writer']
    assert [f'{a.llm.provider}/{a.llm.model}' for a in crew.agents] == [SEARCH_MODEL, model, model]
    assert crew.process is Process.sequential and len(crew.tasks) == 3
    assert [t.agent for t in crew.tasks] == crew.agents
    assert crew.tasks[1].context == [crew.tasks[0]]
    assert crew.tasks[2].context == crew.tasks[:2]
    assert crew.agents[0].tools and not crew.agents[1].tools and not crew.agents[2].tools
    assert all(a.llm.max_tokens == 2048 and not a.allow_delegation for a in crew.agents)
    assert '[Document: notes.md]' in crew.tasks[1].description
    assert '[Document: notes.md]' in crew.tasks[2].description


def test_both_routes_receive_identical_evidence_and_prompts():
    crews = [agents.create_research_crew(SIMPLE, document_context='[Document: notes.md] evidence',
                                        synthesis_model=model)
             for model in (FAST_SYNTHESIS_MODEL, QUALITY_SYNTHESIS_MODEL)]
    assert [t.description for t in crews[0].tasks] == [t.description for t in crews[1].tasks]
    assert [t.expected_output for t in crews[0].tasks] == [t.expected_output for t in crews[1].tasks]


def test_auto_scope_matches_selected_manual_route():
    assert scope().digest == scope(mode='fast').digest
    assert scope(COMPLEX).digest == scope(COMPLEX, 'quality').digest
    assert scope().digest != scope(mode='quality').digest
    original = scope()
    for field, value in [
        ('router_policy_version', 'v2'), ('selected_route', 'quality'),
        ('search_model', 'other'), ('synthesis_model', 'other'), ('version', '1'),
    ]:
        assert replace(original, **{field: value}).digest != original.digest


def test_cross_route_cache_misses_without_embeddings(tmp_path):
    provider = FakeEmbeddingProvider()
    cache = SemanticCacheService(provider, ChromaCacheStore(tmp_path / 'cache'))
    cache.save(SIMPLE, 'fast answer', scope(mode='fast'))
    with patch.object(provider, 'embed_text', wraps=provider.embed_text) as embed:
        assert cache.lookup(SIMPLE, scope(mode='quality')).hit is None
        assert cache.lookup('Explain MCP.', scope(mode='quality')).hit is None
        embed.assert_not_called()
    cache.save(SIMPLE, 'quality answer', scope(mode='quality'))
    assert cache.lookup(SIMPLE, scope()).hit.entry.answer == 'fast answer'
    assert cache.lookup(SIMPLE, scope(mode='quality')).hit.entry.answer == 'quality answer'


def test_actual_phase6_record_safely_misses_and_is_retained(tmp_path):
    # Write the old persisted JSON/digest, not merely a v2 record marked v1.
    cache = SemanticCacheService(FakeEmbeddingProvider(), ChromaCacheStore(tmp_path / 'cache'),
                                 clock=lambda: 1000.)
    state = asdict(CacheScope(use_rag=False))
    for key in ('router_policy_version', 'selected_route', 'search_model', 'synthesis_model'):
        state.pop(key)
    state.update(version='1', research_model=QUALITY_SYNTHESIS_MODEL)
    digest_state = dict(state)
    digest_state['corpus_fingerprint'] = 'WEB_ONLY'
    for key in ('rag_top_k', 'rag_max_distance', 'max_context_chars', 'max_chunk_chars', 'rag_embedding_model'):
        digest_state.pop(key)
    old_digest = digest(digest_state)
    old_id = digest([old_digest, normalize_query(SIMPLE)])
    cache.store._collection.upsert(
        ids=[old_id], documents=['phase6 answer'], embeddings=[[1., 0., .1]],
        metadatas=[dict(query=SIMPLE, normalized_query=normalize_query(SIMPLE),
                        created_at=1000., expires_at=4600., scope=old_digest,
                        scope_json=json.dumps(state))],
    )
    with patch.object(cache.embedding_provider, 'embed_text') as embed:
        assert cache.lookup(SIMPLE, scope()).hit is None
        assert cache.lookup(SIMPLE, scope(mode='quality')).hit is None
        embed.assert_not_called()
    assert cache.store.count() == 1


@pytest.mark.parametrize('mode,query', [('auto', SIMPLE), ('auto', COMPLEX), ('fast', COMPLEX), ('quality', SIMPLE)])
def test_cache_hit_bypasses_concrete_runtime_boundaries(tmp_path, mode, query):
    cache = SemanticCacheService(FakeEmbeddingProvider(), ChromaCacheStore(tmp_path / 'cache'))
    cache.save(query, 'cached answer', scope(query, mode))
    with patch('crewai.crew.Crew.kickoff') as kickoff, patch.object(ConcreteDDGS, 'text') as ddgs, \
         patch('agents.retrieve_document_context') as retrieve, patch('agents.create_research_crew') as factory:
        assert agents.run_research(query, use_rag=False, model_route=mode, cache_service=cache) == 'cached answer'
    for spy in (kickoff, ddgs, retrieve, factory):
        spy.assert_not_called()


def test_router_precedes_scope_and_lookup_and_write_reuses_decision(tmp_path):
    events = []
    original_router, original_scope = agents.route_query, agents.research_cache_scope
    cache = SemanticCacheService(FakeEmbeddingProvider(), ChromaCacheStore(tmp_path / 'cache'))
    original_lookup = cache.lookup

    def routed(*args, **kwargs):
        events.append('router')
        return original_router(*args, **kwargs)

    def scoped(*args, **kwargs):
        events.append('scope')
        return original_scope(*args, **kwargs)

    def lookup(*args, **kwargs):
        events.append('lookup')
        return original_lookup(*args, **kwargs)

    crew = Mock(kickoff=Mock(return_value=SimpleNamespace(raw='answer')))
    with patch('agents.route_query', side_effect=routed) as router, \
         patch('agents.research_cache_scope', side_effect=scoped), \
         patch.object(cache, 'lookup', side_effect=lookup), \
         patch('agents.create_research_crew', return_value=crew) as factory:
        assert agents.run_research(COMPLEX, use_rag=False, cache_service=cache) == 'answer'
    router.assert_called_once_with(COMPLEX, 'auto')
    assert events == ['router', 'scope', 'lookup', 'scope']
    factory.assert_called_once_with(COMPLEX, document_context='', synthesis_model=QUALITY_SYNTHESIS_MODEL)
    assert cache.lookup(COMPLEX, scope(COMPLEX)).hit.entry.answer == 'answer'


@pytest.mark.parametrize('mode,model', [('auto', FAST_SYNTHESIS_MODEL),
                                      ('fast', FAST_SYNTHESIS_MODEL), ('quality', QUALITY_SYNTHESIS_MODEL)])
def test_disabled_cache_still_routes_once_and_does_no_cache_or_rag_work(mode, model):
    cache = Mock()
    crew = Mock(kickoff=Mock(return_value=SimpleNamespace(raw='answer')))
    with patch('agents.get_default_cache_service') as default, patch('agents.research_cache_scope') as scoped, \
         patch('agents.get_default_rag_service') as rag, \
         patch('agents.create_research_crew', return_value=crew) as factory:
        assert agents.run_research(SIMPLE, use_rag=False, use_cache=False, cache_service=cache,
                                   model_route=mode) == 'answer'
    factory.assert_called_once_with(SIMPLE, document_context='', synthesis_model=model)
    crew.kickoff.assert_called_once_with()
    for spy in (default, scoped, rag):
        spy.assert_not_called()
    assert not cache.mock_calls


def test_invalid_route_fails_before_expensive_work():
    with patch('agents.get_default_rag_service') as rag, patch('agents.create_research_crew') as crew:
        assert agents.run_research(SIMPLE, model_route='invalid').startswith('Error:')
    rag.assert_not_called()
    crew.assert_not_called()
