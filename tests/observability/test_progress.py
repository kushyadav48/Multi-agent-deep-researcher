"""Deterministic progress boundaries. No model or network requests."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from unittest.mock import Mock, patch

import pytest
from crewai import Agent
from ddgs.ddgs import DDGS as ConcreteDDGS
from pydantic import ValidationError

import agents
from observability.progress import (
    AGENT_STAGES, ExecutionProgressEvent, ExecutionStage as S, StageStatus as T, ProgressEmitter,
)
from rag.models import RetrievedChunk
from semantic_cache.models import CacheLookup
from tests.observability.test_execution import QUERY, RESULTS, crew_output
from tests.ui.fixtures import execution_result


def event_pairs(events):
    return [(event.stage, event.status) for event in events]


def fake_kickoff(crew):
    output = crew_output()
    crew.agents[0].tools[0].to_structured_tool().func(query=QUERY)
    for task, result in zip(crew.tasks, output.tasks_output):
        task.output = result
        if task.callback:
            returned = task.callback(result)
            if asyncio.iscoroutine(returned):
                asyncio.run(returned)
        if crew.task_callback and crew.task_callback != task.callback:
            crew.task_callback(result)
    return output


def run(events=None, **kwargs):
    options = dict(use_rag=False, use_cache=False, record_metrics=False)
    options.update(kwargs)
    with patch('crewai.crew.Crew.kickoff', autospec=True, side_effect=fake_kickoff), \
         patch.object(ConcreteDDGS, 'text', return_value=RESULTS) as ddgs:
        result = agents.run_research_detailed(QUERY,
            progress_callback=events.append if events is not None else None, **options)
    return result, ddgs


@pytest.mark.parametrize('stage', list(S))
@pytest.mark.parametrize('status', list(T))
def test_typed_event_roundtrip(stage, status):
    event = ExecutionProgressEvent(request_id='request', stage=stage, status=status,
                                   message='completed', metadata={'result_count': 2}, elapsed_ms=1)
    assert ExecutionProgressEvent.model_validate_json(event.model_dump_json()) == event
    assert event.timestamp.endswith('+00:00')
    assert not {'embeddings', 'reasoning', 'messages', 'system_prompt'} & type(event).model_fields.keys()


@pytest.mark.parametrize('kwargs', [
    {'stage': 'THOUGHT'}, {'status': 'THINKING'}, {'elapsed_ms': -1}, {'elapsed_ms': float('inf')},
    {'embeddings': [1., 2.]}, {'reasoning': 'PRIVATE'}, {'system_prompt': 'PRIVATE'},
    {'metadata': {'embeddings': [1., 2.]}}, {'metadata': {'messages': 'PRIVATE'}},
    {'metadata': {'query': {'scratchpad': 'PRIVATE'}}}, {'metadata': {'result_count': [1]}},
])
def test_rejects_unsupported_or_private_fields(kwargs):
    values = dict(request_id='request', stage=S.ROUTING, status=T.STARTED, message='running...')
    values.update(kwargs)
    with pytest.raises(ValidationError):
        ExecutionProgressEvent(**values)


def test_normal_sequence_real_ddgs_boundary_no_duplicate_work_and_no_callback_equivalence(capsys):
    events = []
    result, ddgs = run(events)
    old, old_ddgs = run()
    ddgs.assert_called_once_with(QUERY, max_results=5)
    old_ddgs.assert_called_once()
    assert event_pairs(events) == [
        (S.ROUTING, T.STARTED), (S.ROUTING, T.COMPLETED),
        (S.CACHE, T.STARTED), (S.CACHE, T.COMPLETED), (S.RAG, T.SKIPPED),
        (S.WEB_SEARCHER, T.STARTED), (S.WEB_SEARCH, T.STARTED), (S.WEB_SEARCH, T.COMPLETED),
        (S.WEB_SEARCHER, T.COMPLETED), (S.ANALYST, T.STARTED), (S.ANALYST, T.COMPLETED),
        (S.WRITER, T.STARTED), (S.WRITER, T.COMPLETED),
        (S.FINALIZING, T.STARTED), (S.FINALIZING, T.COMPLETED), (S.COMPLETE, T.COMPLETED),
    ]
    assert result.final_answer == old.final_answer and result.warnings == old.warnings == []
    assert result.status == old.status == 'SUCCESS'
    assert [a.output_text for a in result.agents] == [a.output_text for a in old.agents]
    assert all(e.request_id == result.request_id for e in events)
    routing = events[1]
    assert routing.metadata == dict(route=result.routing.selected_route, complexity=result.routing.complexity,
                                    score=result.routing.score, threshold=result.routing.threshold)
    assert next(e for e in events if e.stage == S.WEB_SEARCH and e.status == T.COMPLETED).metadata['result_count'] == 2
    assert all(e.elapsed_ms is None for e in events if e.stage in AGENT_STAGES)
    assert 'PRIVATE' not in json.dumps([e.model_dump(mode='json') for e in events])
    assert capsys.readouterr().out == ''


def test_router_runs_once_and_callback_failure_is_safe():
    bad = Mock(side_effect=RuntimeError('SECRET credential traceback'))
    with patch('agents.route_query', wraps=agents.route_query) as router:
        with patch('crewai.crew.Crew.kickoff', return_value=crew_output()):
            result = agents.run_research_detailed(QUERY, use_rag=False, use_cache=False,
                record_metrics=False, progress_callback=bad)
    router.assert_called_once_with(QUERY, agents.RoutingMode.AUTO)
    bad.assert_called_once()
    assert result.status == 'SUCCESS' and result.final_answer == 'writer deliverable'
    assert result.warnings == ['Progress callback failed (RuntimeError).']
    assert 'SECRET' not in json.dumps(result.to_dict())


@pytest.mark.parametrize('kind', ['exact_hit', 'semantic_hit'])
def test_cache_hit_skips_all_work(kind):
    fixture = execution_result(kind)
    hit = Mock(kind='exact' if kind == 'exact_hit' else 'semantic', similarity=.99,
               entry=Mock(id='entry', answer='cached answer'))
    cache = Mock(lookup=Mock(return_value=CacheLookup(hit=hit)))
    events = []
    with patch('agents.create_research_crew') as factory, patch.object(ConcreteDDGS, 'text') as ddgs, \
         patch('agents.retrieve_document_context') as retrieval:
        result = agents.run_research_detailed(QUERY, use_rag=False, cache_service=cache,
                                              record_metrics=False, progress_callback=events.append)
    factory.assert_not_called()
    ddgs.assert_not_called()
    retrieval.assert_not_called()
    cache.lookup.assert_called_once()
    assert events[3].metadata['cache_status'] == fixture.cache.status
    assert event_pairs(events)[4:9] == [(stage, T.SKIPPED) for stage in
        (S.RAG, S.WEB_SEARCH, S.WEB_SEARCHER, S.ANALYST, S.WRITER)]
    assert event_pairs(events)[-3:] == [(S.FINALIZING, T.STARTED), (S.FINALIZING, T.COMPLETED), (S.COMPLETE, T.COMPLETED)]
    assert all(a.output_text is None for a in result.agents)


@pytest.mark.parametrize('lookup,status', [(CacheLookup(), 'EMPTY'), (CacheLookup(embedding=(1., 0.)), 'MISS')])
def test_real_cache_status_from_single_lookup(lookup, status):
    events = []
    cache = Mock(lookup=Mock(return_value=lookup))
    result, _ = run(events, use_cache=True, cache_service=cache)
    cache.lookup.assert_called_once()
    assert events[3].message == result.cache.status == status


@pytest.mark.parametrize('count,chunks,status', [
    (0, [], 'EMPTY_CORPUS'), (1, [], 'NO_RELEVANT_RESULTS'),
    (1, [RetrievedChunk('PRIVATE local text', 'notes.md', None, 0, .2)], 'USED'),
])
def test_rag_events_follow_actual_evidence(count, chunks, status):
    events = []
    service = Mock(count=Mock(return_value=count), retrieve=Mock(return_value=chunks))
    result, _ = run(events, use_rag=True, rag_service=service)
    rag = [e for e in events if e.stage == S.RAG]
    assert [e.status for e in rag] == [T.STARTED, T.COMPLETED]
    assert rag[-1].metadata == dict(chunk_count=result.rag.chunk_count, retrieval_status=status)
    assert rag[-1].elapsed_ms == result.rag.duration_ms
    assert 'PRIVATE' not in json.dumps([e.model_dump(mode='json') for e in events])
    assert service.retrieve.call_count == (1 if count else 0)


@pytest.mark.parametrize('construction', [False, True])
def test_rag_error_keeps_fallback(construction):
    events = []
    service = Mock(count=Mock(return_value=1), retrieve=Mock(side_effect=OSError('SECRET')))
    with patch('agents.get_default_rag_service', return_value=service,
               side_effect=OSError('SECRET') if construction else None):
        result, _ = run(events, use_rag=True)
    assert result.status == 'SUCCESS' and 'web-only research' in result.final_answer
    assert (S.RAG, T.ERROR) in event_pairs(events)
    assert 'SECRET' not in json.dumps([e.model_dump(mode='json') for e in events])


@pytest.mark.parametrize('error', [None, OSError('SECRET network details')])
def test_empty_or_failed_web_is_truthful_and_keeps_fallback(error):
    events = []
    with patch('crewai.crew.Crew.kickoff', autospec=True, side_effect=fake_kickoff), \
         patch.object(ConcreteDDGS, 'text', return_value=[], side_effect=error) as ddgs:
        result = agents.run_research_detailed(QUERY, use_rag=False, use_cache=False,
                                              record_metrics=False, progress_callback=events.append)
    ddgs.assert_called_once()
    web = [e for e in events if e.stage == S.WEB_SEARCH]
    assert [e.status for e in web] == [T.STARTED, T.ERROR if error else T.COMPLETED]
    assert web[-1].metadata['result_count'] == 0 and web[-1].elapsed_ms > 0
    assert result.status == 'SUCCESS'
    assert 'SECRET' not in json.dumps([e.model_dump(mode='json') for e in events])


def test_routing_and_crew_failures_preserve_external_errors():
    for route, expected_stage in [('invalid', S.ROUTING), ('auto', S.WEB_SEARCHER)]:
        events = []
        with patch('crewai.crew.Crew.kickoff', side_effect=RuntimeError('SECRET')):
            result = agents.run_research_detailed(QUERY, model_route=route, use_rag=False,
                use_cache=False, record_metrics=False, progress_callback=events.append)
        assert result.status == 'ERROR' and result.final_answer.startswith('Error:')
        assert (expected_stage, T.ERROR) in event_pairs(events)
        assert event_pairs(events)[-1] == (S.COMPLETE, T.ERROR)
        assert 'SECRET' not in json.dumps([e.model_dump(mode='json') for e in events])


def test_public_task_execute_sync_drives_callbacks_and_preserves_existing_callbacks():
    # Real public Task execution with a fake Agent executor verifies the installed
    # callback contract rather than just making a fake call the callback itself.
    crew = agents.create_research_crew(QUERY)
    events = []
    emitter = ProgressEmitter(execution_result(), events.append)
    previous = Mock()
    crew.tasks[0].callback = previous
    crew_callback = Mock()
    crew.task_callback = crew_callback
    emitter.attach_crew(crew)
    for agent in crew.agents:
        agent.crew = crew
    with patch.object(Agent, 'execute_task', return_value='safe deliverable') as executor:
        for task in crew.tasks:
            task.execute_sync()
    assert executor.call_count == 3
    previous.assert_called_once_with(crew.tasks[0].output)
    assert crew_callback.call_count == 3
    assert event_pairs(events) == [(S.WEB_SEARCHER, T.COMPLETED), (S.ANALYST, T.STARTED),
        (S.ANALYST, T.COMPLETED), (S.WRITER, T.STARTED), (S.WRITER, T.COMPLETED)]


def test_shared_task_crew_callback_is_not_duplicated_and_async_is_preserved():
    crew = agents.create_research_crew(QUERY)
    shared = Mock()
    crew.task_callback = crew.tasks[0].callback = shared
    called = []

    async def prior(output):
        called.append(output.raw)

    crew.tasks[1].callback = prior
    emitter = ProgressEmitter(execution_result(), lambda event: None)
    emitter.attach_crew(crew)
    for agent in crew.agents:
        agent.crew = crew
    with patch.object(Agent, 'execute_task', return_value='safe deliverable'):
        for task in crew.tasks:
            task.execute_sync()
    assert shared.call_count == 3 and called == ['safe deliverable']


def test_no_ddgs_invocation_and_incomplete_public_outputs_are_not_fabricated():
    events = []
    output = crew_output()
    output.tasks_output = output.tasks_output[:1]
    with patch('crewai.crew.Crew.kickoff', return_value=output):
        result = agents.run_research_detailed(QUERY, use_rag=False, use_cache=False,
                                              record_metrics=False, progress_callback=events.append)
    assert (S.WEB_SEARCH, T.SKIPPED) in event_pairs(events)
    assert (S.ANALYST, T.COMPLETED) not in event_pairs(events)
    assert (S.WRITER, T.COMPLETED) not in event_pairs(events)
    assert result.agents[1].output_text is None


def test_finalizing_includes_cache_save_and_metrics_persistence():
    events = []
    cache = Mock(lookup=Mock(return_value=CacheLookup()))
    store = Mock()

    def save(*args, **kwargs):
        assert event_pairs(events)[-1] == (S.FINALIZING, T.STARTED)

    def record(result):
        assert event_pairs(events)[-1] == (S.FINALIZING, T.STARTED)

    cache.save.side_effect = save
    store.record.side_effect = record
    result, _ = run(events, use_cache=True, cache_service=cache, metrics_store=store, record_metrics=True)
    cache.save.assert_called_once()
    store.record.assert_called_once()
    assert result.metrics_persisted
    assert event_pairs(events)[-1] == (S.COMPLETE, T.COMPLETED)


def test_concurrent_callback_failure_warns_once_without_stdout(capsys):
    result = execution_result()
    callback = Mock(side_effect=OSError('SECRET'))
    emitter = ProgressEmitter(result, callback)
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(lambda _: emitter.emit(S.WEB_SEARCH, T.STARTED, 'running...'), range(8)))
    callback.assert_called_once()
    assert result.warnings == ['Progress callback failed (OSError).']
    assert capsys.readouterr().out == ''


def test_concurrent_tool_events_are_request_local_and_match_actual_calls():
    events = []
    emitter = ProgressEmitter(execution_result(), events.append)
    tool = agents.DuckDuckGoSearchTool()
    tool.observe_progress(emitter)

    def search(query, **kwargs):
        return RESULTS[:1] if query == 'one' else RESULTS

    with patch.object(ConcreteDDGS, 'text', side_effect=search) as ddgs, \
         ThreadPoolExecutor(max_workers=2) as workers:
        list(workers.map(tool._run, ('one', 'two')))
    assert ddgs.call_count == 2
    completed = {e.metadata['query']: e for e in events if e.status == T.COMPLETED}
    assert completed['one'].metadata['result_count'] == 1
    assert completed['two'].metadata['result_count'] == 2
    for call in tool.search_calls:
        assert completed[call.query].elapsed_ms == call.duration_ms


def test_cache_error_and_writer_failure_identify_safe_stages():
    events = []
    cache = Mock(lookup=Mock(side_effect=OSError('SECRET')))
    result, _ = run(events, use_cache=True, cache_service=cache)
    assert result.status == 'SUCCESS' and (S.CACHE, T.ERROR) in event_pairs(events)
    events.clear()

    def fail_writer(crew):
        for task, output in zip(crew.tasks[:2], crew_output().tasks_output[:2]):
            task.output = output
            task.callback(output)
        raise RuntimeError('SECRET')

    with patch('crewai.crew.Crew.kickoff', autospec=True, side_effect=fail_writer):
        result = agents.run_research_detailed(QUERY, use_rag=False, use_cache=False,
                                              record_metrics=False, progress_callback=events.append)
    assert result.status == 'ERROR' and result.final_answer == 'Error: SECRET'
    assert (S.WRITER, T.ERROR) in event_pairs(events)
    assert [agent.status for agent in result.agents] == ['COMPLETED', 'COMPLETED', 'ERROR']
    assert 'SECRET' not in json.dumps([e.model_dump(mode='json') for e in events])


@pytest.mark.parametrize('answer', ['', 'Error: SECRET'])
def test_invalid_final_response_reports_finalization_failure(answer):
    events = []
    with patch('crewai.crew.Crew.kickoff', return_value=crew_output(answer)):
        result = agents.run_research_detailed(QUERY, use_rag=False, use_cache=False,
                                              record_metrics=False, progress_callback=events.append)
    assert result.final_answer == answer and result.status == 'ERROR'
    assert (S.FINALIZING, T.ERROR) in event_pairs(events)
    assert event_pairs(events)[-1] == (S.COMPLETE, T.ERROR)
    assert 'SECRET' not in json.dumps([e.model_dump(mode='json') for e in events])
