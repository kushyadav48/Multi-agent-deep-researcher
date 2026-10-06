import json
import subprocess
import sys

import pytest

from routing import ModelRoute, RoutingMode, route_query
from routing.models import FAST_SYNTHESIS_MODEL, QUALITY_SYNTHESIS_MODEL, SEARCH_MODEL, ROUTER_POLICY_VERSION
from routing.router import QUALITY_THRESHOLD


SIMPLE = 'What is the Model Context Protocol (MCP)? Give a concise explanation.'
COMPLEX = ('Compare MCP with REST APIs, analyze security risks and trade-offs, '
           'and recommend an enterprise architecture.')


@pytest.mark.parametrize('query', [
    'What is MCP?', 'What does RAG mean?', 'Explain RAG.',
    'Explain semantic caching.', 'Summarize MCP in three bullets.',
    'What is architecture?', SIMPLE,
])
def test_simple_requests_use_fast(query):
    decision = route_query(query)
    assert decision.selected_route is ModelRoute.FAST
    assert decision.score < QUALITY_THRESHOLD
    assert decision.search_model == SEARCH_MODEL
    assert decision.synthesis_model == FAST_SYNTHESIS_MODEL


@pytest.mark.parametrize('query', [
    'Compare MCP with REST APIs and explain trade-offs.',
    'Critique this proposal.', 'Evaluate evidence quality.',
    'Recommend an architecture.', 'Design an enterprise multi-agent system.',
    'Compare these documents, identify contradictions, evaluate evidence quality, '
    'and recommend the strongest conclusion.',
    COMPLEX,
])
def test_analytical_requests_use_quality(query):
    decision = route_query(query)
    assert decision.selected_route is ModelRoute.QUALITY
    assert decision.score >= QUALITY_THRESHOLD
    assert decision.search_model == SEARCH_MODEL
    assert decision.synthesis_model == QUALITY_SYNTHESIS_MODEL


def test_decision_reasons_and_policy_are_deterministic():
    decision = route_query(COMPLEX)
    assert all(route_query(COMPLEX) == decision for _ in range(100))
    assert decision.policy_version == ROUTER_POLICY_VERSION == 'v1'
    assert decision.reasons == (
        'Comparison requested (+2)',
        'Trade-offs or pros/cons requested (+2)',
        'Evaluation, critique, or risk analysis requested (+3)',
        'Recommendation or architecture design requested (+3)',
        'Multiple requested actions (+1)',
        'Auto: score 11 >= 3; selected quality',
    )
    assert route_query('What is MCP?').reasons == ('Auto: score 0 < 3; selected fast',)


@pytest.mark.parametrize('mode,query,route,model', [
    (RoutingMode.FAST, COMPLEX, ModelRoute.FAST, FAST_SYNTHESIS_MODEL),
    (' FAST ', COMPLEX, ModelRoute.FAST, FAST_SYNTHESIS_MODEL),
    (RoutingMode.QUALITY, SIMPLE, ModelRoute.QUALITY, QUALITY_SYNTHESIS_MODEL),
    ('QUALITY', SIMPLE, ModelRoute.QUALITY, QUALITY_SYNTHESIS_MODEL),
])
def test_manual_override(mode, query, route, model):
    decision = route_query(query, mode)
    assert decision.selected_route is route
    assert decision.synthesis_model == model and decision.search_model == SEARCH_MODEL
    assert decision.reasons[-1] == f'Manual {route.value} override'
    assert decision.score == route_query(query).score


def test_length_score_is_bounded_and_categories_count_once():
    assert route_query('word ' * 29).score == 0
    assert route_query('word ' * 30).score == 1
    assert route_query('word ' * 80).score == 2
    assert route_query('compare compare compare').score == 2
    assert route_query('word ' * 80 + ' compare A and B').selected_route is ModelRoute.QUALITY


@pytest.mark.parametrize('query,mode', [('', 'auto'), ('  ', 'auto'), (None, 'auto'), ('MCP', 'unknown')])
def test_invalid_input(query, mode):
    with pytest.raises(ValueError):
        route_query(query, mode)


def test_router_import_and_execution_have_no_inference_dependencies():
    # Fresh process catches even accidental import-time model/embedding work.
    script = """
import builtins, json
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'crewai', 'ollama', 'rag', 'chromadb', 'litellm', 'agents'}:
        raise AssertionError('Router imported an inference dependency: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from routing import route_query
print(json.dumps(route_query('What is MCP?').selected_route.value))
"""
    result = subprocess.check_output([sys.executable, '-c', script], text=True)
    assert json.loads(result) == 'fast'
