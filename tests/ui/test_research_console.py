from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from observability.models import ResearchExecutionResult, Usage, WebSearchCall
from tests.ui.fixtures import execution_result


def replay(result):
    app = AppTest.from_string(
        "import streamlit as st\nfrom ui.research_console import render_research_console\n"
        "render_research_console(st.session_state.execution)", default_timeout=30,
    )
    app.session_state.execution = result
    app.run()
    assert not app.exception
    return app


def visible(app):
    return "\n".join(element.value for kind in ("text", "caption", "markdown", "warning", "error", "info")
                     for element in app.get(kind))


@pytest.mark.parametrize("kind,classification,route", [
    ("simple", "SIMPLE", "FAST"), ("complex", "COMPLEX", "QUALITY"),
])
def test_summary_actual_models_public_outputs_and_open_final(kind, classification, route):
    result = execution_result(kind)
    with patch("agents.run_research_detailed") as research, patch("agents.DDGS") as ddgs:
        app = replay(result)
    research.assert_not_called()
    ddgs.assert_not_called()
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Request Type"] == classification and metrics["Selected Route"] == route
    assert metrics["Cache"] == "MISS" and metrics["Total runtime"] == "1.60 s"
    assert metrics["Web Search"] == "USED" and metrics["Web results"] == "1"
    assert metrics["Tokens"] == "30"
    text = visible(app)
    for model in (result.routing.search_model, result.routing.analyst_model, result.routing.writer_model):
        assert model in text
    for name, output in zip(("Web Searcher", "Research Analyst", "Technical Writer"), result.agents):
        section = next(section for section in app.expander if section.label == f"{name} Output")
        assert section.markdown[0].value == output.output_text
    assert any(header.value == "Final Research Answer" for header in app.subheader)
    assert result.final_answer in [item.value for item in app.markdown]
    assert all(result.final_answer not in [item.value for item in section.markdown] for section in app.expander)
    assert all(not section.proto.expanded for section in app.expander)
    assert "Quality threshold: 3" in text and "Router policy version: v1" in text
    assert result.routing.reasons[0] in text


def test_rag_evidence_full_text_metadata_without_fake_page():
    result = execution_result("complex")
    app = replay(result)
    text = visible(app)
    assert "Source: architecture.pdf" in text and "Source: notes.md" in text
    assert [item.value for item in app.text if item.value.startswith("Page:")] == ["Page: 6"]
    assert "Chunk index: 3" in text and "Distance: 0.180" in text
    assert result.rag.evidence[0].text.rstrip() in text and result.rag.evidence[1].text in text
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Local RAG"] == "USED" and metrics["RAG chunks"] == "2"


def test_multiple_ddgs_calls_remain_separate_and_links_are_safe():
    result = execution_result()
    result.web.calls.append(WebSearchCall("Second distinct query", [], 25, "EMPTY"))
    app = replay(result)
    text = visible(app)
    assert "Search Call 1" in text and "Search Call 2" in text
    for value in ("MCP public protocol", "Second distinct query", "MCP documentation", "Open protocol snippet"):
        assert value in text
    assert app.get("link_button")[0].proto.url == "https://modelcontextprotocol.io/"
    result.web.calls[0].results[0].url = "javascript:alert(1)"
    app = replay(result)
    assert not app.get("link_button")
    assert "URL: javascript:alert(1)" in visible(app)


@pytest.mark.parametrize("kind,status", [("semantic_hit", "SEMANTIC HIT"), ("exact_hit", "EXACT HIT")])
def test_cache_hit_honest_bypass_and_nullable_usage(kind, status):
    app = replay(execution_result(kind))
    text = visible(app)
    assert {metric.label: metric.value for metric in app.metric}["Cache"] == status
    assert "Full research pipeline bypassed." in text
    assert "Crew: Not executed" in text and "RAG: Not executed" in text and "DDGS: Not executed" in text
    assert f"Not executed — {status.lower()}." in text
    assert "Configured threshold: > 0.97" in text
    assert "Input tokens: Not reported" in text and "Total tokens: Not reported" in text
    assert "Crew: 0" not in text
    assert "Raw writer deliverable" not in text


def test_timings_and_partial_usage_are_not_invented():
    result = execution_result()
    result.usage = Usage(input_tokens=20)
    app = replay(result)
    text = visible(app)
    for value in ("Routing: 1.0 ms", "Cache lookup: 2.0 ms", "RAG retrieval: Not executed",
                  "Web search: 125.0 ms", "Crew: 1.50 s", "Total: 1.60 s",
                  "Input tokens: 20", "Output tokens: Not reported", "Total tokens: Not reported"):
        assert value in text


def test_failure_retains_observed_outputs_and_hides_raw_exception():
    app = replay(execution_result("error"))
    assert app.error and app.warning
    text = visible(app)
    assert "RuntimeError" in text and "Observable search deliverable" in text
    assert "crew failed" in text and "secret credential" not in text and "traceback" not in text


def test_missing_route_outputs_and_details_do_not_crash_or_leak_extra_fields():
    result = ResearchExecutionResult("early-failure", "question", "start", "auto")
    result.system_prompt = "PRIVATE SYSTEM PROMPT"
    result.provider_messages = "PRIVATE PROVIDER MESSAGES"
    result.embeddings = "PRIVATE EMBEDDINGS"
    result.agents[0].scratchpad = "PRIVATE SCRATCHPAD"
    app = replay(result)
    text = visible(app)
    assert "Routing decision: Not available" in text and "Model: Not available" in text
    assert "PRIVATE" not in text
    assert "Metrics persisted: No" in text and "End time (UTC): Not available" in text
