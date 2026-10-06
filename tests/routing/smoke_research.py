"""Opt-in single FAST research regression, with a 20-minute worker bound."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import json
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import agents
from crewai import Process
from crewai.crew import Crew
from crewai.events import crewai_event_bus
from ddgs.ddgs import DDGS as ConcreteDDGS
from routing import ModelRoute, route_query
from routing.models import FAST_SYNTHESIS_MODEL, QUALITY_SYNTHESIS_MODEL, SEARCH_MODEL


QUERY = 'What is the Model Context Protocol (MCP)? Give a concise explanation.'
COMPLEX_QUERY = ('Compare MCP with REST APIs, analyze security risks and trade-offs, '
                 'and recommend an enterprise architecture.')
ROOT = Path(__file__).resolve().parents[2]


def validate_report(report):
    """Validate captured evidence without ever executing research or inference."""
    assert report['routing_decision']['selected_route'] == ModelRoute.FAST
    assert report['agent_models'] == [SEARCH_MODEL, FAST_SYNTHESIS_MODEL, FAST_SYNTHESIS_MODEL]
    assert report['process'] == Process.sequential.value
    assert report['counts']['crew'] == 1
    assert report['counts']['ddgs'] >= 1 and report['retrieved_urls']
    outputs = report['task_outputs']
    assert len(outputs) == 3 and all(output['raw'].strip() for output in outputs)
    assert [output['agent'] for output in outputs] == report['agent_roles']
    answer = report['answer']
    assert answer.strip() and not answer.startswith('Error:')
    cited = set(re.findall(r'https?://[^\s<>\])]+', answer))
    retrieved = set(report['retrieved_urls'])
    report['cited_urls'] = sorted(cited)
    report['unsupported_urls'] = sorted(cited - retrieved)
    assert cited & retrieved, 'Final answer did not cite retrieved evidence'
    assert not report['unsupported_urls'], 'Final answer invented source URLs'
    report['all_agents_completed'] = True
    report['passed'] = True


def run_smoke():
    directory = ROOT / 'validation_logs'
    directory.mkdir(exist_ok=True)
    report_path = directory / 'phase7_research_regression.json'
    decision = route_query(QUERY)
    report = dict(query=QUERY, routing_decision=asdict(decision),
                  use_rag=False, use_cache=False, model_route='auto',
                  crew_target='crewai.crew.Crew.kickoff', ddgs_target='ddgs.ddgs.DDGS.text',
                  counts=dict(crew=0, ddgs=0), retrieved_urls=[], task_outputs=[])

    def checkpoint():
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    checkpoint()
    assert decision.selected_route is ModelRoute.FAST
    quality = route_query(COMPLEX_QUERY)
    quality_crew = agents.create_research_crew(COMPLEX_QUERY, synthesis_model=quality.synthesis_model)
    report['quality_static'] = dict(routing_decision=asdict(quality),
                                   models=[f'{a.llm.provider}/{a.llm.model}' for a in quality_crew.agents], kickoff=False)
    assert quality.selected_route is ModelRoute.QUALITY
    assert report['quality_static']['models'] == [SEARCH_MODEL, QUALITY_SYNTHESIS_MODEL, QUALITY_SYNTHESIS_MODEL]
    assert quality_crew.process is Process.sequential and len(quality_crew.agents) == 3
    checkpoint()
    original_factory, original_kickoff, original_text = agents.create_research_crew, Crew.kickoff, ConcreteDDGS.text
    assert type(agents.DDGS()) is ConcreteDDGS

    def observed_factory(*args, **kwargs):
        crew = original_factory(*args, **kwargs)
        report['agent_models'] = [f'{a.llm.provider}/{a.llm.model}' for a in crew.agents]
        report['agent_roles'] = [a.role for a in crew.agents]
        report['process'] = crew.process.value
        checkpoint()  # Capture actual assignments before generation starts.
        assert report['agent_models'] == [SEARCH_MODEL, FAST_SYNTHESIS_MODEL, FAST_SYNTHESIS_MODEL]
        assert len(crew.agents) == len(crew.tasks) == 3 and crew.process is Process.sequential
        return crew

    def observed_kickoff(*args, **kwargs):
        report['counts']['crew'] += 1
        checkpoint()
        result = original_kickoff(*args, **kwargs)
        report['task_outputs'] = [dict(agent=output.agent, raw=output.raw) for output in result.tasks_output]
        checkpoint()
        return result

    def observed_text(*args, **kwargs):
        report['counts']['ddgs'] += 1
        checkpoint()
        results = original_text(*args, **kwargs)
        report['retrieved_urls'].extend(item['href'] for item in results if item.get('href'))
        checkpoint()
        return results

    started = perf_counter()
    try:
        with (directory / 'phase7_crew.log').open('w', encoding='utf-8') as log, \
             redirect_stdout(log), redirect_stderr(log), \
             patch.object(agents, 'create_research_crew', observed_factory), \
             patch.object(Crew, 'kickoff', observed_kickoff), \
             patch.object(ConcreteDDGS, 'text', observed_text):
            answer = agents.run_research(QUERY, use_cache=False, use_rag=False, model_route='auto')
            report['seconds'] = perf_counter() - started
            report['answer'] = answer
            checkpoint()
            assert crewai_event_bus.flush(timeout=30.0), 'Crew events did not flush'
        validate_report(report)
    except Exception as error:
        report['seconds'] = perf_counter() - started
        report['passed'] = False
        report['failure'] = str(error)
        checkpoint()
        raise
    checkpoint()
    print(json.dumps({key: value for key, value in report.items() if key not in ('answer', 'task_outputs')},
                     indent=2))
    print(f'Report: {report_path}')


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        run_smoke()
    else:
        # Exactly one research execution; never retry because a spy/assertion fails.
        subprocess.run([sys.executable, '-m', 'tests.routing.smoke_research', '--worker'],
                       cwd=ROOT, check=True, timeout=1200)
