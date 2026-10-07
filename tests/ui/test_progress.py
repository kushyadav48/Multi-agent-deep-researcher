"""Replay live stage events in AppTest, then retain the Phase 9 console."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident
from unittest.mock import Mock, patch

import pytest
from streamlit.testing.v1 import AppTest

from observability.progress import ExecutionProgressEvent, ExecutionStage as S, StageStatus as T
from tests.ui.fixtures import execution_result
from ui.progress import ResearchProgress, format_stage


APP = str(Path(__file__).resolve().parents[2] / 'app.py')


def replay_events(result, *, rag_disabled=False, web_error=False):
    def event(stage, status, message, **kwargs):
        return ExecutionProgressEvent(request_id=result.request_id, stage=stage, status=status,
                                      message=message, **kwargs)
    events = [event(S.ROUTING, T.STARTED, 'running...'),
              event(S.ROUTING, T.COMPLETED, result.routing.selected_route.upper()),
              event(S.CACHE, T.STARTED, 'running...'),
              event(S.CACHE, T.COMPLETED, result.cache.status.replace('_', ' '))]
    if result.cache.status in ('EXACT_HIT', 'SEMANTIC_HIT'):
        events.extend(event(stage, T.SKIPPED, 'skipped (cache hit)') for stage in
                      (S.RAG, S.WEB_SEARCH, S.WEB_SEARCHER, S.ANALYST, S.WRITER))
    else:
        if rag_disabled:
            events.append(event(S.RAG, T.SKIPPED, 'skipped (disabled)'))
        else:
            events.extend([event(S.RAG, T.STARTED, 'running...'),
                           event(S.RAG, T.COMPLETED, '2 chunks retrieved')])
        events.extend([event(S.WEB_SEARCHER, T.STARTED, 'running...'),
                       event(S.WEB_SEARCH, T.STARTED, 'running...'),
                       event(S.WEB_SEARCH, T.ERROR if web_error else T.COMPLETED,
                             'search failed; continuing with available evidence' if web_error else '1 results')])
        for current, following in [(S.WEB_SEARCHER, S.ANALYST), (S.ANALYST, S.WRITER)]:
            events.extend([event(current, T.COMPLETED, 'completed'), event(following, T.STARTED, 'running...')])
        events.append(event(S.WRITER, T.COMPLETED, 'completed'))
    events.extend([event(S.FINALIZING, T.STARTED, 'running...'),
                   event(S.FINALIZING, T.COMPLETED, 'completed'),
                   event(S.COMPLETE, T.COMPLETED, 'Research complete')])
    return events


@pytest.mark.parametrize('kind,rag_disabled,web_error', [
    ('complex', False, False), ('semantic_hit', False, False),
    ('simple', True, False), ('simple', True, True),
])
def test_live_progress_console_and_no_recomputation(kind, rag_disabled, web_error):
    result = execution_result(kind)
    if web_error:
        result.web.calls[0].status = 'ERROR'
        result.web.calls[0].results = []
        result.web.calls[0].error_type = 'OSError'
    events = replay_events(result, rag_disabled=rag_disabled, web_error=web_error)
    snapshots = []
    rendered = Event()
    script_threads = set()
    backend_threads = set()
    cache_hit = result.cache.status in ('EXACT_HIT', 'SEMANTIC_HIT')
    original = ResearchProgress.accept

    def accept(panel, event):
        original(panel, event)
        snapshots.append(panel.events.copy())
        script_threads.add(get_ident())
        if event.stage == S.COMPLETE:
            rendered.set()

    def backend(query, *, progress_callback, **kwargs):
        assert callable(progress_callback)
        backend_threads.add(get_ident())
        for event in events:
            if event.stage == S.WEB_SEARCH:
                # Mimic public CrewAI behavior: tools may use additional workers.
                with ThreadPoolExecutor(max_workers=1) as tools:
                    tools.submit(progress_callback, event).result()
            else:
                progress_callback(event)
        # All stage rows have updated before the backend returns the answer.
        assert rendered.wait(timeout=10)
        assert snapshots[-1][S.COMPLETE].status == T.COMPLETED
        return result

    service = Mock(count=Mock(return_value=2))
    with patch('rag.context.get_default_rag_service', return_value=service), \
         patch('agents.run_research_detailed', side_effect=backend) as research, \
         patch.object(ResearchProgress, 'accept', accept):
        app = AppTest.from_file(APP, default_timeout=30).run()
        if rag_disabled:
            app.checkbox(key='use_rag').uncheck().run()
        app.text_area(key='research_query').set_value('question')
        next(button for button in app.button if button.label == 'Research').click().run()
        assert not app.exception
        text = '\n'.join(row.value for row in app.text)
        assert f'✓ Routing — {result.routing.selected_route.upper()}' in text
        assert f'✓ Cache — {result.cache.status.replace("_", " ")}' in text
        assert '✓ Complete — Research complete' in text
        if cache_hit:
            for label in ('RAG', 'Web Search', 'Web Searcher', 'Analyst', 'Writer'):
                assert f'– {label} — skipped (cache hit)' in text
        elif rag_disabled:
            assert '– RAG — skipped (disabled)' in text
        else:
            assert '✓ RAG — 2 chunks retrieved' in text
        if web_error:
            assert '! Web Search — search failed' in text
        statuses = app.get('status')
        assert statuses[0].label == ('Research complete (with warnings)' if web_error else 'Research complete')
        assert result.final_answer in [item.value for item in app.markdown]
        assert sum(item.value == 'Final Research Answer' for item in app.subheader) == 1
        assert len(snapshots) == len(events)
        assert len(script_threads) == 1 and script_threads.isdisjoint(backend_threads)
        writer_running = next((snapshot for snapshot in snapshots if
                               snapshot.get(S.WRITER) and snapshot[S.WRITER].status == T.STARTED), None)
        if not cache_hit:
            assert writer_running is not None and S.COMPLETE not in writer_running
        app.run()
        app.session_state.console_view = 'Analytics'
        app.run()
        app.button(key='refresh_analytics').click().run()
        app.session_state.console_view = 'Research'
        app.run()  # Includes re-rendering all expandable console sections.
        research.assert_called_once()
        assert app.session_state.research_execution is result
        assert not app.exception
        assert not app.get('status')  # The live panel is transient, the result persists.
        assert result.final_answer in [item.value for item in app.markdown]


def test_stage_rows_are_safe_text_and_only_use_available_timing():
    event = ExecutionProgressEvent(request_id='request', stage=S.WEB_SEARCH, status=T.COMPLETED,
                                   message='5 results', elapsed_ms=6000)
    assert format_stage(S.WEB_SEARCH, event) == '✓ Web Search — 5 results · 6.00 s'
    assert format_stage(S.WRITER) == '○ Writer — pending'
    event = event.model_copy(update={'elapsed_ms': None})
    assert ' s' not in format_stage(S.WEB_SEARCH, event)


@pytest.mark.parametrize('raises', [False, True])
def test_error_panel_and_console_hide_raw_details(raises):
    result = execution_result('error')

    def backend(query, *, progress_callback, **kwargs):
        progress_callback(ExecutionProgressEvent(request_id=result.request_id, stage=S.ANALYST,
                           status=T.ERROR, message='failed', metadata={'error_type': 'RuntimeError'}))
        if raises:
            raise RuntimeError('SECRET trace')
        progress_callback(ExecutionProgressEvent(request_id=result.request_id, stage=S.COMPLETE,
                           status=T.ERROR, message='Research failed'))
        return result

    with patch('rag.context.get_default_rag_service', return_value=Mock(count=Mock(return_value=0))), \
         patch('agents.run_research_detailed', side_effect=backend):
        app = AppTest.from_file(APP, default_timeout=30).run()
        app.text_area(key='research_query').set_value('question')
        next(button for button in app.button if button.label == 'Research').click().run()
    assert not app.exception and app.error
    assert app.get('status')[0].label == 'Research failed'
    text = '\n'.join(item.value for kind in ('text', 'markdown', 'error') for item in app.get(kind))
    assert 'SECRET' not in text and 'secret credential' not in text


def test_panel_ignores_events_from_another_request():
    app = AppTest.from_string(
        "from ui.progress import ResearchProgress\n"
        "from observability.progress import ExecutionProgressEvent\n"
        "panel = ResearchProgress()\n"
        "for request in ('one', 'two'):\n"
        "    panel.accept(ExecutionProgressEvent(request_id=request, stage='ROUTING',\n"
        "                 status='COMPLETED', message=request))\n",
    ).run()
    assert not app.exception
    assert '✓ Routing — one' in [row.value for row in app.text]
    assert not any('two' in row.value for row in app.text)


def test_failed_search_remains_visible_after_retry():
    app = AppTest.from_string(
        "from ui.progress import ResearchProgress\n"
        "from observability.progress import ExecutionProgressEvent\n"
        "panel = ResearchProgress()\n"
        "for status in ('ERROR', 'STARTED', 'COMPLETED'):\n"
        "    panel.accept(ExecutionProgressEvent(request_id='one', stage='WEB_SEARCH',\n"
        "                 status=status, message='search status'))\n"
        "panel.finish(True)\n",
    ).run()
    assert not app.exception
    assert 'earlier search failed' in '\n'.join(row.value for row in app.text)
    assert app.get('status')[0].label == 'Research complete (with warnings)'
