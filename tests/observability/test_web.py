from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from ddgs.ddgs import DDGS as ConcreteDDGS

from agents import DuckDuckGoSearchTool
from tests.observability.test_execution import RESULTS


def test_real_adapter_records_same_data_once_and_preserves_search_options():
    tool = DuckDuckGoSearchTool()
    with patch.object(ConcreteDDGS, 'text', return_value=RESULTS) as ddgs:
        answer = tool.to_structured_tool().func(query='actual query', depth='deep')
    ddgs.assert_called_once_with('actual query', max_results=10)
    assert len(tool.search_calls) == 1
    call = tool.search_calls[0]
    assert call.query == 'actual query' and call.duration_ms > 0 and call.status == 'SUCCESS'
    for observed, item in zip(call.results, RESULTS):
        assert (observed.title, observed.url, observed.snippet) == (item['title'], item['href'], item['body'])
        assert all(value in answer for value in (observed.title, observed.url, observed.snippet))


@pytest.mark.parametrize('result,error,status', [([], None, 'EMPTY'), (None, OSError('offline'), 'ERROR')])
def test_empty_and_failed_calls_are_counted(result, error, status):
    tool = DuckDuckGoSearchTool()
    with patch.object(ConcreteDDGS, 'text', return_value=result, side_effect=error) as ddgs:
        answer = tool._run('query')
    ddgs.assert_called_once()
    assert ('Error occurred' if error else 'No search results') in answer
    call = tool.search_calls[0]
    assert call.status == status and call.results == [] and call.duration_ms > 0
    assert call.error_type == ('OSError' if error else None)


def test_missing_snippets_and_repeated_calls():
    tool = DuckDuckGoSearchTool()
    with patch.object(ConcreteDDGS, 'text', return_value=[{'title': 'Title', 'href': 'https://example.com/'}]):
        tool._run('one')
        tool._run('two')
    assert [call.query for call in tool.search_calls] == ['one', 'two']
    assert tool.search_calls[0].results[0].snippet == ''
    assert tool.retrieved_urls == ('https://example.com/',)


def test_concurrent_requests_do_not_share_trace():
    tools = [DuckDuckGoSearchTool(), DuckDuckGoSearchTool()]
    with patch.object(ConcreteDDGS, 'text', return_value=RESULTS):
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda pair: pair[1]._run(str(pair[0])), enumerate(tools)))
    assert [tool.search_calls[0].query for tool in tools] == ['0', '1']
    assert all(len(tool.search_calls) == 1 for tool in tools)
    assert DuckDuckGoSearchTool().search_calls == ()
