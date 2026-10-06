from unittest.mock import Mock, patch

import pytest
from streamlit.testing.v1 import AppTest

from observability.store import MetricsStore
from tests.ui.fixtures import execution_result


def replay(store):
    with patch("observability.store.get_default_metrics_store", return_value=store):
        app = AppTest.from_string("from ui.analytics import render_analytics\nrender_analytics()",
                                  default_timeout=30).run()
    assert not app.exception
    return app


def test_real_sqlite_summary_routes_evidence_latency_and_private_recent_table(tmp_path):
    store = MetricsStore(tmp_path / "metrics.db")
    for index, kind in enumerate(("simple", "complex", "semantic_hit", "error")):
        result = execution_result(kind)
        result.query = "DO NOT SHOW PRIVATE QUERY OR CONTENT"
        result.timings.total_ms = (index + 1) * 1000
        store.record(result)
    app = replay(store)
    metrics = {item.label: item.value for item in app.metric}
    assert metrics == {
        "Total Requests": "4", "Successful Requests": "3", "Failed Requests": "1",
        "Cache Hit Rate": "25.0%", "Average Latency": "2.50 s",
        "P50 Latency": "2.50 s", "P95 Latency": "3.85 s",
        "Cache hits": "1", "Cache lookups": "4", "Requests using RAG": "1",
        "Requests using DDGS": "3", "DDGS calls": "3",
    }
    route_table, recent_table = [item.value for item in app.markdown]
    assert "| FAST | 3 |" in route_table and "| QUALITY | 1 |" in route_table
    lines = recent_table.splitlines()
    assert len(lines) == 6  # Header, separator, four rows.
    assert "ERROR" in lines[2] and "4.00 s" in lines[2]
    assert all(len(line.split("|")[2].strip()) == 10 for line in lines[2:])
    assert "Not reported" in recent_table
    assert "PRIVATE" not in recent_table
    assert "DO NOT SHOW" not in str(app)
    assert not any(name in lines[0] for name in ("raw_query", "final_answer", "output_text", "snippet"))


def test_empty_sqlite_has_clear_state_and_no_broken_charts(tmp_path):
    app = replay(MetricsStore(tmp_path / "empty.db"))
    assert any("No research metrics have been recorded yet." == item.value for item in app.info)
    assert {item.label: item.value for item in app.metric}["Total Requests"] == "0"
    assert not app.dataframe and not app.get("arrow_vega_lite_chart")


@pytest.mark.parametrize("failure", ["construction", "summary", "recent"])
def test_read_failure_is_safe(failure):
    store = Mock()
    if failure != "construction":
        getattr(store, failure).side_effect = OSError("SECRET credential database path")
    with patch("observability.store.get_default_metrics_store", return_value=store,
               side_effect=OSError("SECRET") if failure == "construction" else None):
        app = AppTest.from_string("from ui.analytics import render_analytics\nrender_analytics()").run()
    assert not app.exception and app.warning
    assert "SECRET" not in app.warning[0].value


def test_unmeasured_historical_latency_does_not_draw_chart(tmp_path):
    store = MetricsStore(tmp_path / "no_latency.db")
    result = execution_result()
    result.timings.total_ms = None
    result.cache.status = "DISABLED"
    store.record(result)
    app = replay(store)
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Cache Hit Rate"] == metrics["Average Latency"] == "Not available"
    assert not app.get("arrow_vega_lite_chart")
