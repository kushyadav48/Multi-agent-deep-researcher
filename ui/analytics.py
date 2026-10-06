"""Operational history through MetricsStore's public API only."""

import streamlit as st

from observability import store as metrics_store
from ui.formatting import duration, markdown_table, rate, recent_rows


def render_analytics():
    st.subheader("Analytics")
    st.caption("Operational history only. Queries, answers, agent output, and evidence are not stored in analytics.")
    st.button("Refresh Analytics", key="refresh_analytics", icon=":material/refresh:")
    try:
        store = metrics_store.get_default_metrics_store()
        summary = store.summary()
        recent = store.recent(limit=20)
    except Exception:
        st.warning("Research metrics are unavailable. Research remains available; try refreshing analytics later.")
        return

    fields = [
        ("Total Requests", summary["total_requests"]),
        ("Successful Requests", summary["successful_requests"]),
        ("Failed Requests", summary["failed_requests"]),
        ("Cache Hit Rate", rate(summary["cache_hit_rate"])),
    ]
    for column, (name, value) in zip(st.columns(4), fields):
        column.metric(name, value, border=True)
    for column, (name, key) in zip(st.columns(3), [
        ("Average Latency", "average_latency_ms"),
        ("P50 Latency", "p50_latency_ms"), ("P95 Latency", "p95_latency_ms"),
    ]):
        column.metric(name, duration(summary[key], missing="Not available"), border=True)
    if summary["total_requests"] == 0:
        st.info("No research metrics have been recorded yet.")
        return

    route_column, cache_column = st.columns(2)
    with route_column, st.container(border=True):
        st.subheader("Route Usage")
        st.markdown(markdown_table([
            {"Route": "FAST", "Requests": summary["fast_count"]},
            {"Route": "QUALITY", "Requests": summary["quality_count"]},
        ]))
    with cache_column, st.container(border=True):
        st.subheader("Cache Usage")
        st.metric("Cache hits", summary["cache_hits"])
        st.metric("Cache lookups", summary["cache_lookups"])
        st.caption("Hit rate uses eligible lookups (EMPTY, MISS, EXACT HIT, SEMANTIC HIT). Disabled, bypassed, and failed lookups are excluded.")

    with st.container(border=True):
        st.subheader("Evidence Usage")
        for column, (name, key) in zip(st.columns(3), [
            ("Requests using RAG", "rag_usage_count"),
            ("Requests using DDGS", "ddgs_usage_count"),
            ("DDGS calls", "ddgs_call_count"),
        ]):
            column.metric(name, summary[key])

    st.subheader("Recent Requests")
    st.caption("Latest 20 requests, newest first. Query identity is a 10-character hash prefix; timestamps are UTC.")
    st.markdown(markdown_table(recent_rows(recent)))
    st.caption("Summary cards cover all recorded requests; recent requests show at most 20. The Total Time column contains each measured latency.")
