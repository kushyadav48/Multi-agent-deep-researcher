"""Render only public fields of the current ResearchExecutionResult."""

from urllib.parse import urlsplit

import streamlit as st

from observability.models import ResearchExecutionResult
from ui.formatting import duration, label, optional, reason, tokens


def render_research_console(result: ResearchExecutionResult):
    route = result.routing
    cache_hit = result.cache.status in ("EXACT_HIT", "SEMANTIC_HIT")
    st.subheader("Question")
    st.text(result.query)

    with st.container(border=True):
        st.subheader("Execution Summary")
        st.caption(f"Request status: {result.status}")
        fields = [
            ("Request Type", route.complexity if route else "Not available"),
            ("Selected Route", label(route.selected_route if route else None)),
            ("Cache", label(result.cache.status)),
            ("Local RAG", "USED" if result.rag.used else
             "DISABLED" if result.rag.status == "DISABLED" else "NOT USED"),
            ("Web Search", "USED" if result.web.used else "NOT USED"),
            ("RAG chunks", str(result.rag.chunk_count)),
            ("Web results", str(result.web.total_results)),
            ("Total runtime", duration(result.timings.total_ms, missing="Not available")),
            ("Tokens", tokens(result.usage.total_tokens)),
        ]
        for start in range(0, len(fields), 3):
            for column, (name, value) in zip(st.columns(3), fields[start:start + 3]):
                column.metric(name, value)
        for column, (name, model) in zip(st.columns(3), [
            ("Searcher", route.search_model if route else None),
            ("Analyst", route.analyst_model if route else None),
            ("Writer", route.writer_model if route else None),
        ]):
            column.caption(name)
            column.text(optional(model))
        st.caption("Model assignments describe the selected route; execution status is shown below.")

    for warning in result.warnings:
        st.warning(warning)
    if result.status == "ERROR":
        # The backend's error answer can contain arbitrary exception text.
        # Its error_type is the public exception class / response category.
        st.error("Research could not be completed. Check Ollama and your internet connection, then try again.")
        st.caption(f"Error type: {optional(result.error_type)}")

    with st.expander("Routing Decision"):
        if route is None:
            st.caption("Routing decision: Not available")
        else:
            for name, value in [
                ("Requested mode", label(route.requested_mode)),
                ("Selected route", label(route.selected_route)),
                ("Request Type", route.complexity),
                ("Complexity score", route.score),
                ("Quality threshold", route.threshold),
                ("Router policy version", route.policy_version),
            ]:
                st.text(f"{name}: {value}")
            st.caption("Routing reasons")
            for explanation in route.reasons:
                st.text(f"• {explanation}")

    with st.expander("Semantic Cache"):
        st.text(f"Status: {label(result.cache.status)}")
        if result.cache.similarity is not None:
            st.text(f"Similarity: {result.cache.similarity:.3f}")
        if result.cache.distance is not None:
            st.text(f"Distance: {result.cache.distance:.3f}")
        threshold = result.cache.threshold
        st.text(f"Configured threshold: > {threshold:g}" if threshold is not None
                else "Configured threshold: Not available")
        if result.cache.reason:
            st.text(f"Reason: {reason(result.cache.reason)}")
        if cache_hit:
            st.info("Full research pipeline bypassed.")
            st.text("Crew: Not executed\nDDGS: Not executed\nRAG: Not executed")

    with st.expander("RAG Evidence"):
        st.text(f"Status: {label(result.rag.status)}")
        st.text(f"Retrieved chunks: {result.rag.chunk_count}")
        st.text(f"Retrieval time: {duration(result.rag.duration_ms)}")
        if result.rag.reason:
            st.text(f"Reason: {reason(result.rag.reason)}")
        if not result.rag.evidence:
            st.caption("No local evidence was used for this request.")
        for index, item in enumerate(result.rag.evidence, 1):
            with st.container(border=True):
                st.text(f"Chunk {index} · Source: {item.source}")
                if item.page is not None:
                    st.text(f"Page: {item.page}")
                st.text(f"Chunk index: {item.chunk_index}")
                if item.distance is not None:
                    st.text(f"Distance: {item.distance:.3f}")
                with st.expander(f"Retrieved text · Chunk {index}"):
                    st.text(item.text)

    with st.expander("Web Search Results"):
        if not result.web.calls:
            st.caption("DDGS: Not executed")
        for index, call in enumerate(result.web.calls, 1):
            with st.container(border=True):
                st.markdown(f"**Search Call {index}**")
                st.text(f"Search query: {call.query}")
                st.caption(f"Status: {call.status} · Duration: {duration(call.duration_ms)} · "
                           f"Returned results: {len(call.results)}")
                if call.error_type:
                    st.text(f"Error type: {call.error_type}")
                if not call.results:
                    st.caption("No results returned.")
                for item in call.results:
                    st.text(item.title or "Untitled result")
                    # Limit links to real web schemes; never render raw HTML.
                    try:
                        url = urlsplit(item.url)
                        safe_url = url.scheme in ("http", "https") and bool(url.netloc)
                    except ValueError:
                        safe_url = False
                    if safe_url:
                        st.link_button(item.url, item.url)
                    else:
                        st.text(f"URL: {item.url or 'Not available'}")
                    if item.snippet:
                        st.text(item.snippet)

    st.caption("Agent content below is observable task output. Hidden reasoning and provider messages are never shown.")
    for name, task in [("Web Searcher", "web_search"),
                       ("Research Analyst", "analysis"), ("Technical Writer", "writing")]:
        with st.expander(f"{name} Output"):
            agent = next((item for item in result.agents if item.task_name == task), None)
            if agent is None:
                st.caption("Task output: Not available")
                continue
            st.text(f"Status: {label(agent.status)}")
            st.text(f"Model: {optional(agent.model)}")
            if agent.output_text is not None:
                st.markdown(agent.output_text)
            elif cache_hit and agent.status == "NOT_EXECUTED":
                st.caption(f"Not executed — {label(result.cache.status).lower()}.")
            else:
                st.caption(f"{label(agent.status).capitalize()} — {reason(agent.reason)}.")

    with st.expander("Execution Timings"):
        for name, value in [
            ("Routing", result.timings.routing_ms),
            ("Cache lookup", result.timings.cache_lookup_ms),
            ("RAG retrieval", result.timings.rag_retrieval_ms),
            ("Web search", result.timings.web_search_ms),
            ("Crew", result.timings.crew_ms),
            ("Total", result.timings.total_ms),
        ]:
            st.text(f"{name}: {duration(value)}")
        st.caption("Web search time is included in Crew time; stage timings are not additive.")
        for name, value in [("Input tokens", result.usage.input_tokens),
                            ("Output tokens", result.usage.output_tokens),
                            ("Total tokens", result.usage.total_tokens)]:
            st.text(f"{name}: {tokens(value)}")

    with st.expander("Request Details"):
        st.text(f"Request ID: {result.request_id}")
        st.text(f"Start time (UTC): {result.started_at}")
        st.text(f"End time (UTC): {optional(result.finished_at)}")
        st.text(f"Metrics persisted: {'Yes' if result.metrics_persisted else 'No'}")

    with st.container(border=True):
        st.subheader("Final Research Answer")
        if result.status == "ERROR":
            st.caption("No completed research answer is available for this failed request.")
        elif result.final_answer:
            st.markdown(result.final_answer)
        else:
            st.caption("No final answer was reported.")
