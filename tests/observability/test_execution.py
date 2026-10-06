from dataclasses import fields
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from crewai.crews.crew_output import CrewOutput
from crewai.tasks.task_output import TaskOutput
from crewai.types.usage_metrics import UsageMetrics
from ddgs.ddgs import DDGS as ConcreteDDGS

import agents
from observability.models import RAGEvidence, ResearchExecutionResult
from observability.store import MetricsStore
from rag.models import RetrievedChunk
from routing.models import FAST_SYNTHESIS_MODEL, SEARCH_MODEL
from semantic_cache.models import CacheLookup
from semantic_cache.service import SemanticCacheService
from semantic_cache.store import ChromaCacheStore
from tests.rag.helpers import FakeEmbeddingProvider


QUERY = 'What is MCP?'
RESULTS = [dict(title='MCP', href='https://modelcontextprotocol.io/', body='Open protocol'),
           dict(title='Tools', href='https://example.com/tools', body='Tools and data')]


def crew_output(raw='writer deliverable', *, usage=None):
    return CrewOutput(raw=raw, tasks_output=[
        TaskOutput(description='PRIVATE SYSTEM PROMPT', messages=[{'role': 'system', 'content': 'PRIVATE SCRATCHPAD'}],
                   agent=role, raw=text)
        for role, text in zip(('Web Searcher', 'Research Analyst', 'Technical Writer'),
                              ('search deliverable', 'analysis deliverable', raw))
    ], token_usage=usage or UsageMetrics())


def run_fake(**kwargs):
    with patch('crewai.crew.Crew.kickoff', return_value=crew_output()):
        return agents.run_research_detailed(QUERY, use_cache=False, use_rag=False, **kwargs)


def test_public_outputs_order_models_and_safe_serialization(tmp_path):
    store = MetricsStore(tmp_path / 'metrics.db')
    result = run_fake(metrics_store=store)
    assert isinstance(result, ResearchExecutionResult)
    assert result.status == 'SUCCESS' and result.metrics_persisted
    assert [agent.output_text for agent in result.agents] == [
        'search deliverable', 'analysis deliverable', 'writer deliverable']
    assert [agent.model for agent in result.agents] == [SEARCH_MODEL, FAST_SYNTHESIS_MODEL, FAST_SYNTHESIS_MODEL]
    assert all(agent.status == 'COMPLETED' for agent in result.agents)
    assert result.routing.requested_mode == 'auto' and result.routing.selected_route == 'fast'
    assert result.routing.complexity == 'SIMPLE' and result.routing.threshold == 3
    assert result.routing.policy_version == 'v1' and result.routing.reasons
    assert result.cache.status == result.rag.status == 'DISABLED'
    assert result.timings.total_ms > 0 and result.timings.crew_ms > 0
    assert result.timings.routing_ms > 0
    assert result.timings.cache_lookup_ms is result.timings.rag_retrieval_ms is result.timings.web_search_ms is None
    assert result.started_at.endswith('+00:00') and result.finished_at.endswith('+00:00')
    assert result.request_id and result.final_answer == 'writer deliverable'
    assert result.usage.total_tokens is None
    serialized = json.dumps(result.to_dict())
    assert 'PRIVATE' not in serialized and 'embedding' not in serialized
    assert 'embedding' not in {field.name for field in fields(RAGEvidence)}
    assert result.to_dict()['web']['total_results'] == 0
    assert store.count() == 1


def test_wrapper_is_string_and_delegates_all_options():
    detailed = SimpleNamespace(final_answer='answer')
    with patch('agents.run_research_detailed', return_value=detailed) as backend:
        answer = agents.run_research(QUERY, use_rag=False, use_cache=False, model_route='quality')
    assert type(answer) is str and answer == 'answer'
    assert backend.call_args.kwargs['model_route'] == 'quality'
    assert not backend.call_args.kwargs['use_rag'] and not backend.call_args.kwargs['use_cache']


def test_writer_is_distinct_from_postprocessed_answer():
    def kickoff(crew):
        crew.agents[0].tools[0].to_structured_tool().func(query=QUERY)
        return crew_output()
    with patch('crewai.crew.Crew.kickoff', autospec=True, side_effect=kickoff), \
         patch.object(ConcreteDDGS, 'text', return_value=RESULTS) as search:
        result = agents.run_research_detailed(QUERY, use_rag=False, use_cache=False, record_metrics=False)
    search.assert_called_once_with(QUERY, max_results=5)
    assert result.agents[2].output_text == 'writer deliverable'
    assert result.final_answer.startswith('writer deliverable\n\nSources retrieved:')
    assert all(item['href'] in result.final_answer for item in RESULTS)
    assert result.web.total_calls == 1 and result.web.total_results == 2
    assert result.timings.web_search_ms == result.web.calls[0].duration_ms > 0
    assert not result.metrics_persisted


@pytest.mark.parametrize('usage,expected', [
    (UsageMetrics(prompt_tokens=20, completion_tokens=10, total_tokens=30), (20, 10, 30)),
    (UsageMetrics(), (None, None, None)),
    (UsageMetrics(prompt_tokens=20, completion_tokens=10, total_tokens=999), (None, None, None)),
])
def test_public_usage_only_if_measured_and_consistent(usage, expected):
    with patch('crewai.crew.Crew.kickoff', return_value=crew_output(usage=usage)):
        result = agents.run_research_detailed(QUERY, use_rag=False, use_cache=False, record_metrics=False)
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.total_tokens) == expected


@pytest.mark.parametrize('mode,complexity,model', [
    ('fast', 'SIMPLE', FAST_SYNTHESIS_MODEL), ('quality', 'COMPLEX', SEARCH_MODEL),
])
def test_actual_decision_reused_once(mode, complexity, model):
    with patch('agents.route_query', wraps=agents.route_query) as router:
        result = run_fake(model_route=mode, record_metrics=False)
    router.assert_called_once_with(QUERY, mode)
    assert result.routing.complexity == complexity
    assert result.agents[1].model == result.agents[2].model == model


@pytest.mark.parametrize('chunks,count,status', [
    ([], 0, 'EMPTY_CORPUS'), ([], 1, 'NO_RELEVANT_RESULTS'),
    ([RetrievedChunk('far', 'notes.md', None, 0, .9)], 1, 'NO_RELEVANT_RESULTS'),
    ([RetrievedChunk('local evidence', 'notes.pdf', 3, 7, .2)], 1, 'USED'),
    ([RetrievedChunk('nonpaged', 'notes.md', None, 2, .1)], 1, 'USED'),
])
def test_rag_observes_original_retrieval(chunks, count, status):
    rag = Mock(count=Mock(return_value=count), retrieve=Mock(return_value=chunks))
    with patch('crewai.crew.Crew.kickoff', return_value=crew_output()):
        result = agents.run_research_detailed(QUERY, rag_service=rag, use_cache=False, record_metrics=False)
    assert result.rag.status == status and result.rag.enabled
    assert result.rag.duration_ms == result.timings.rag_retrieval_ms > 0
    if status == 'USED':
        assert result.rag.used and result.rag.chunk_count == 1
        evidence = result.rag.evidence[0]
        chunk = chunks[0]
        assert (evidence.source, evidence.page, evidence.chunk_index, evidence.distance, evidence.text) == (
            chunk.source, chunk.page, chunk.chunk_index, chunk.distance, chunk.text)
    else:
        assert not result.rag.used and result.rag.evidence == []
    if count:
        rag.retrieve.assert_called_once_with(QUERY, top_k=4)
    else:
        rag.retrieve.assert_not_called()


@pytest.mark.parametrize('construction', [False, True])
def test_rag_failure_preserves_web_fallback(construction):
    rag = Mock(count=Mock(return_value=1), retrieve=Mock(side_effect=RuntimeError('offline')))
    with patch('crewai.crew.Crew.kickoff', return_value=crew_output()), \
         patch('agents.get_default_rag_service', side_effect=RuntimeError('offline') if construction else None,
               return_value=rag):
        result = agents.run_research_detailed(QUERY, use_cache=False, record_metrics=False)
    assert result.status == 'SUCCESS' and result.rag.status == 'ERROR'
    assert not result.rag.used and not result.rag.evidence
    assert 'web-only research' in result.final_answer and result.warnings
    assert (result.timings.rag_retrieval_ms is None) == construction


@pytest.mark.parametrize('kind', ['exact', 'semantic'])
def test_cache_hits_bypass_all_runtime_stages_and_persist(tmp_path, kind):
    cache = SemanticCacheService(FakeEmbeddingProvider(), ChromaCacheStore(tmp_path / 'cache'))
    rag = Mock(corpus_fingerprint=Mock(return_value='corpus'))
    scope = agents.research_cache_scope(rag, use_rag=True, top_k=4, max_distance=.6,
                                      routing_decision=agents.route_query(QUERY))
    cache.save(QUERY, 'cached answer', scope)
    query = QUERY if kind == 'exact' else 'Explain MCP.'
    store = MetricsStore(tmp_path / 'metrics.db')
    with patch('crewai.crew.Crew.kickoff') as kickoff, patch.object(ConcreteDDGS, 'text') as ddgs, \
         patch('agents.retrieve_document_context') as retrieve, patch('agents.create_research_crew') as factory, \
         patch.object(cache, 'lookup', wraps=cache.lookup) as lookup:
        result = agents.run_research_detailed(query, rag_service=rag, cache_service=cache, metrics_store=store)
    lookup.assert_called_once()
    for spy in (kickoff, ddgs, retrieve, factory, rag.retrieve, rag.count):
        spy.assert_not_called()
    assert result.cache.status == ('EXACT_HIT' if kind == 'exact' else 'SEMANTIC_HIT')
    assert result.cache.threshold == .97 and result.cache.entry_id
    assert result.cache.distance == 1 - result.cache.similarity
    assert result.final_answer == 'cached answer'
    assert result.rag.status == 'BYPASSED' and not result.rag.used
    assert all(a.status == 'NOT_EXECUTED' and a.output_text is None and a.reason == 'semantic_cache_hit'
               for a in result.agents)
    assert not result.web.used and result.web.total_results == result.web.total_calls == 0
    assert result.timings.crew_ms is result.timings.web_search_ms is result.timings.rag_retrieval_ms is None
    row = store.recent()[0]
    assert row['cache_status'] == result.cache.status and row['crew_ms'] is None
    assert row['ddgs_call_count'] == row['web_result_count'] == row['rag_used'] == 0
    assert result.metrics_persisted and store.count() == 1


@pytest.mark.parametrize('lookup,status', [(CacheLookup(), 'EMPTY'), (CacheLookup(embedding=(1., 0.)), 'MISS')])
def test_cache_miss_status_without_extra_lookup(lookup, status):
    cache = Mock(lookup=Mock(return_value=lookup))
    with patch('crewai.crew.Crew.kickoff', return_value=crew_output()):
        result = agents.run_research_detailed(QUERY, use_rag=False, cache_service=cache, record_metrics=False)
    assert result.cache.status == status
    cache.lookup.assert_called_once()
    cache.save.assert_called_once()


def test_cache_failure_is_explicit_and_does_not_change_answer():
    cache = Mock(lookup=Mock(side_effect=OSError('private details')))
    with patch('crewai.crew.Crew.kickoff', return_value=crew_output()):
        result = agents.run_research_detailed(QUERY, use_rag=False, cache_service=cache, record_metrics=False)
    assert result.cache.status == 'ERROR' and result.warnings
    assert result.status == 'SUCCESS' and result.final_answer == 'writer deliverable'


def test_metrics_write_and_construction_failure_do_not_destroy_answer(capsys):
    for constructor in (False, True):
        store = Mock(record=Mock(side_effect=OSError('secret credential')))
        with patch('observability.store.get_default_metrics_store', side_effect=OSError('secret credential')):
            result = run_fake(metrics_store=None if constructor else store)
        assert result.status == 'SUCCESS' and result.final_answer == 'writer deliverable'
        assert not result.metrics_persisted
        assert result.warnings == ['Metrics persistence failed (OSError).']
    assert capsys.readouterr().out == ''


def test_failure_keeps_real_web_and_completed_task_outputs(tmp_path):
    def fail(crew):
        crew.agents[0].tools[0]._run(QUERY)
        crew.tasks[0].output = crew_output().tasks_output[0]
        raise RuntimeError('generation failed')
    store = MetricsStore(tmp_path / 'metrics.db')
    with patch('crewai.crew.Crew.kickoff', autospec=True, side_effect=fail), \
         patch.object(ConcreteDDGS, 'text', return_value=RESULTS):
        result = agents.run_research_detailed(QUERY, use_rag=False, use_cache=False, metrics_store=store)
    assert result.final_answer == 'Error: generation failed' and result.status == 'ERROR'
    assert result.agents[0].status == 'COMPLETED' and result.agents[1].status == 'ERROR'
    assert result.web.total_results == 2 and result.timings.crew_ms > 0
    row = store.recent()[0]
    assert row['status'] == 'ERROR' and row['error_type'] == 'RuntimeError'
    assert row['web_result_count'] == 2 and row['total_ms'] > 0


def test_invalid_route_records_failed_request_before_crew(tmp_path):
    store = MetricsStore(tmp_path / 'metrics.db')
    with patch('crewai.crew.Crew.kickoff') as kickoff:
        result = agents.run_research_detailed(QUERY, model_route='invalid', metrics_store=store)
    kickoff.assert_not_called()
    assert result.status == 'ERROR' and result.routing is None
    assert result.final_answer.startswith('Error:')
    assert store.recent()[0]['error_type'] == 'ValueError'


def test_incomplete_outputs_are_reported_without_fabrication():
    with patch('crewai.crew.Crew.kickoff', return_value=SimpleNamespace(raw='answer',
            tasks_output=[SimpleNamespace(raw='search output')])):
        result = agents.run_research_detailed(QUERY, use_rag=False, use_cache=False, record_metrics=False)
    assert result.warnings and result.agents[0].status == 'COMPLETED'
    assert all(a.status == 'UNAVAILABLE' and a.output_text is None for a in result.agents[1:])


@pytest.mark.parametrize('answer,error_type', [
    ('', 'EmptyResearchResponse'), ('Error: failed', 'ResearchResponseError'),
    ('Error occurred while searching: offline', 'ResearchResponseError'),
])
def test_returned_failure_strings_keep_external_behavior_and_failure_metrics(tmp_path, answer, error_type):
    store = MetricsStore(tmp_path / 'metrics.db')
    with patch('crewai.crew.Crew.kickoff', return_value=crew_output(answer)):
        result = agents.run_research_detailed(QUERY, use_rag=False, use_cache=False, metrics_store=store)
    assert result.final_answer == answer and result.status == 'ERROR'
    assert store.recent()[0]['error_type'] == error_type
    assert store.recent()[0]['status'] == 'ERROR' and store.count() == 1


def test_opt_out_does_no_metrics_construction_or_write():
    store = Mock()
    with patch('observability.store.get_default_metrics_store') as default:
        result = run_fake(metrics_store=store, record_metrics=False)
    assert not result.metrics_persisted and result.final_answer == 'writer deliverable'
    default.assert_not_called()
    store.record.assert_not_called()
