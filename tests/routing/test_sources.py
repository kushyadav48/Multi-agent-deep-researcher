from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from ddgs.ddgs import DDGS as ConcreteDDGS

import agents
from routing.models import FAST_SYNTHESIS_MODEL, SEARCH_MODEL


QUERY = 'What is MCP?'
URL = 'https://modelcontextprotocol.io/introduction'
OTHER = 'https://www.ibm.com/think/topics/model-context-protocol'
RESULTS = [dict(title='MCP', href=URL, body='An open protocol connecting AI to tools.'),
           dict(title='MCP', href=OTHER, body='A standard interface.')]


def retrieved_crew():
    crew = agents.create_research_crew(QUERY, synthesis_model=FAST_SYNTHESIS_MODEL)
    # Exercise CrewAI's actual adapter, which retains the original tool callback.
    tool = crew.agents[0].tools[0]
    with patch.object(ConcreteDDGS, 'text', return_value=RESULTS):
        tool.to_structured_tool().func(query=QUERY)
    return crew


def test_tool_retains_exact_deduplicated_urls_and_request_isolation():
    crew = retrieved_crew()
    tool = crew.agents[0].tools[0]
    with patch.object(ConcreteDDGS, 'text', return_value=RESULTS + [dict(href=None)]):
        tool._run(QUERY)
    assert tool.retrieved_urls == (URL, OTHER)
    assert agents.DuckDuckGoSearchTool().retrieved_urls == ()
    assert crew.tasks[0].tools[0] is tool


def test_missing_links_are_preserved_without_inference_or_search():
    crew = retrieved_crew()
    with patch('crewai.crew.Crew.kickoff') as kickoff, patch.object(ConcreteDDGS, 'text') as search:
        answer = agents._with_search_sources('MCP connects AI to external tools.', crew)
    assert answer == f'MCP connects AI to external tools.\n\nSources retrieved:\n- <{URL}>\n- <{OTHER}>'
    kickoff.assert_not_called()
    search.assert_not_called()
    assert agents._with_search_sources(answer, crew) == answer


def test_existing_links_and_local_citations_are_preserved():
    crew = retrieved_crew()
    original = f'MCP connects tools. [Source]({URL}) [Document: notes.md]'
    answer = agents._with_search_sources(original, crew)
    assert answer.startswith(original)
    assert answer.count(URL) == 1 and answer.count(OTHER) == 1
    complete = original + f' [Source]({OTHER})'
    assert agents._with_search_sources(complete, crew) == complete


@pytest.mark.parametrize('answer', ['', 'Error: failed', 'Error occurred while searching: offline'])
def test_errors_and_empty_answers_are_not_expanded(answer):
    assert agents._with_search_sources(answer, retrieved_crew()) == answer


def test_failed_or_empty_search_adds_no_sources():
    for result, error in [([], None), (None, RuntimeError('offline'))]:
        crew = agents.create_research_crew(QUERY, synthesis_model=FAST_SYNTHESIS_MODEL)
        with patch.object(ConcreteDDGS, 'text', return_value=result, side_effect=error):
            crew.agents[0].tools[0]._run(QUERY)
        assert agents._with_search_sources('Search unavailable.', crew) == 'Search unavailable.'


def test_pipeline_returns_and_caches_the_same_source_preserving_answer():
    cache = Mock()
    from semantic_cache.models import CacheLookup
    cache.lookup.return_value = CacheLookup()
    calls = []

    def completed(crew):
        calls.append([f'{a.llm.provider}/{a.llm.model}' for a in crew.agents])
        crew.agents[0].tools[0].to_structured_tool().func(query=QUERY)
        return SimpleNamespace(raw='MCP connects AI to tools.',
                               tasks_output=[SimpleNamespace(raw='complete')] * 3)

    with patch('crewai.crew.Crew.kickoff', autospec=True, side_effect=completed) as kickoff, \
         patch.object(ConcreteDDGS, 'text', return_value=RESULTS) as search:
        answer = agents.run_research(QUERY, use_rag=False, cache_service=cache)
    kickoff.assert_called_once()
    search.assert_called_once()
    assert calls == [[SEARCH_MODEL, FAST_SYNTHESIS_MODEL, FAST_SYNTHESIS_MODEL]]
    assert URL in answer and OTHER in answer
    assert cache.save.call_args.args[1] == answer
