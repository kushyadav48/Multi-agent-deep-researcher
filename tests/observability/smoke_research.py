"""Opt-in single live observability request; never retries generation."""

from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import agents
from crewai.crew import Crew
from crewai.events import crewai_event_bus
from ddgs.ddgs import DDGS as ConcreteDDGS
from observability.store import MetricsStore
from routing.models import FAST_SYNTHESIS_MODEL, SEARCH_MODEL


ROOT = Path(__file__).resolve().parents[2]
QUERY = 'What is the Model Context Protocol (MCP)? Give a concise explanation.'
REPORT_PATH = ROOT / 'validation_logs' / 'phase8_research_regression.json'


def validate_report(report):
    result = report['execution']
    route = result['routing']
    assert result['status'] == 'SUCCESS' and result['metrics_persisted']
    assert route['requested_mode'] == 'auto' and route['selected_route'] == 'fast'
    assert route['complexity'] == 'SIMPLE' and route['score'] < route['threshold']
    assert [route['search_model'], route['analyst_model'], route['writer_model']] == [
        SEARCH_MODEL, FAST_SYNTHESIS_MODEL, FAST_SYNTHESIS_MODEL]
    assert result['cache']['status'] == result['rag']['status'] == 'DISABLED'
    assert report['crew_calls'] == 1
    external = report['ddgs_calls']
    calls = result['web']['calls']
    assert len(external) == len(calls) == result['web']['total_calls'] > 0
    assert result['web']['total_results'] == sum(len(call['results']) for call in external) > 0
    for actual, recorded in zip(external, calls):
        assert actual['query'] == recorded['query']
        assert actual['results'] == recorded['results']
        assert recorded['duration_ms'] > 0
    assert any(item['url'].startswith(('https://', 'http://')) for call in calls for item in call['results'])
    assert len(result['agents']) == 3 and all(
        agent['status'] == 'COMPLETED' and agent['output_text'].strip() for agent in result['agents'])
    assert [agent['output_text'] for agent in result['agents']] == report['public_task_outputs']
    assert result['agents'][2]['output_text'] == report['writer_raw']
    assert result['final_answer'].strip() and not result['final_answer'].startswith('Error:')
    timings = result['timings']
    assert all(timings[key] > 0 for key in ('routing_ms', 'crew_ms', 'web_search_ms', 'total_ms'))
    assert timings['cache_lookup_ms'] is timings['rag_retrieval_ms'] is None
    assert report['metrics_count'] == 1 and report['raw_query_absent']
    row = report['metrics_row']
    assert row['status'] == 'SUCCESS' and row['selected_route'] == 'fast'
    assert row['cache_status'] == 'DISABLED' and row['rag_enabled'] == row['rag_used'] == row['rag_chunk_count'] == 0
    assert row['ddgs_call_count'] == len(calls) and row['web_result_count'] == result['web']['total_results']
    for key in ('search_model', 'analyst_model', 'writer_model'):
        assert row[key] == route[key]
    for key, value in timings.items():
        assert row[key] == value


def run_smoke():
    REPORT_PATH.parent.mkdir(exist_ok=True)
    report = dict(query=QUERY, crew_calls=0, ddgs_calls=[],
                  crew_boundary='crewai.crew.Crew.kickoff', ddgs_boundary='ddgs.ddgs.DDGS.text')

    def checkpoint():
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')

    original_kickoff, original_text = Crew.kickoff, ConcreteDDGS.text

    def kickoff(*args, **kwargs):
        report['crew_calls'] += 1
        checkpoint()
        output = original_kickoff(*args, **kwargs)
        # Copy only returned task deliverables, never TaskOutput.messages.
        report['public_task_outputs'] = [task.raw for task in output.tasks_output]
        report['writer_raw'] = output.tasks_output[2].raw
        checkpoint()
        return output

    def text(*args, **kwargs):
        call = dict(query=args[1] if len(args) > 1 else kwargs['query'], results=[])
        report['ddgs_calls'].append(call)
        checkpoint()
        results = original_text(*args, **kwargs)
        call['results'] = [dict(title=item.get('title') or '', url=item.get('href') or '',
                                snippet=item.get('body') or '') for item in results]
        checkpoint()
        return results

    checkpoint()
    try:
        with TemporaryDirectory(prefix='phase8_metrics_') as directory:
            store = MetricsStore(Path(directory) / 'metrics.db')
            # Discard verbose console rendering; the public outputs are enough.
            with open(os.devnull, 'w', encoding='utf-8') as sink, redirect_stdout(sink), redirect_stderr(sink), \
                 patch.object(Crew, 'kickoff', kickoff), patch.object(ConcreteDDGS, 'text', text):
                result = agents.run_research_detailed(
                    QUERY, use_cache=False, use_rag=False, model_route='auto', metrics_store=store,
                )
                report['execution'] = result.to_dict()
                checkpoint()
                assert crewai_event_bus.flush(timeout=30), 'Crew events did not drain'
            report['metrics_count'] = store.count()
            report['metrics_row'] = store.recent()[0]
            report['metrics_summary'] = store.summary()
            report['raw_query_absent'] = QUERY not in json.dumps(report['metrics_row'])
            # Inspect actual database contents, including the schema.
            with store._connection() as connection:
                report['raw_query_absent'] &= QUERY not in '\n'.join(connection.iterdump())
            checkpoint()
            validate_report(report)
            report['passed'] = True
    except Exception as error:
        report['passed'] = False
        report['failure'] = str(error)
        checkpoint()
        raise
    checkpoint()
    print(json.dumps(dict(passed=True, crew_calls=report['crew_calls'],
                          ddgs_calls=len(report['ddgs_calls']), web_results=result.web.total_results,
                          routing=result.to_dict()['routing'], timings=result.to_dict()['timings'],
                          usage=result.to_dict()['usage'], metrics_count=report['metrics_count'],
                          outputs_captured=[bool(agent.output_text) for agent in result.agents],
                          final_answer_captured=bool(result.final_answer), report=str(REPORT_PATH)), indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        run_smoke()
    else:
        subprocess.run([sys.executable, '-m', 'tests.observability.smoke_research', '--worker'],
                       cwd=ROOT, check=True, timeout=1200)
