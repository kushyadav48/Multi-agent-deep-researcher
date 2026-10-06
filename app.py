"""Streamlit frontend for the canonical research backend."""

import streamlit as st
from pathlib import Path
from tempfile import TemporaryDirectory

from agents import run_research_detailed
from rag.context import get_default_rag_service, source_name
from ui.analytics import render_analytics
from ui.research_console import render_research_console


def knowledge_base_controls():
    """Render corpus controls; only the explicit button ingests documents."""
    with st.sidebar:
        st.subheader("Knowledge Base")
        use_rag = st.checkbox("Use local knowledge base", value=True, key="use_rag")
        uploads = st.file_uploader(
            "PDF, TXT, or Markdown documents", type=["pdf", "txt", "md"],
            accept_multiple_files=True, key="knowledge_files",
        )
        add = st.button("Add to Knowledge Base", key="add_documents")
        try:
            service = get_default_rag_service()
        except Exception:
            st.warning("Knowledge base unavailable. Check local storage and try again.")
            return use_rag
        if add:
            if not uploads:
                st.warning("Choose documents to add first.")
            for upload in uploads or []:
                name = source_name(upload.name)
                try:
                    # Keep raw uploads only while the loader reads them. Stable
                    # source identity keeps citations and repeat upserts useful.
                    with TemporaryDirectory(prefix="research_upload_") as directory:
                        path = Path(directory) / name
                        path.write_bytes(upload.getvalue())
                        with st.spinner(f"Adding {name}..."):
                            count = service.ingest_file(path, source=name)
                    st.success(f"{name}: processed {count} chunks.")
                except Exception:
                    st.error(f"Could not add {name}. Check the file and local embedding service.")
        try:
            st.caption(f"Stored chunks: {service.count()}")
        except Exception:
            st.warning("Could not read corpus status. Check local storage.")
    return use_rag


def main():
    st.set_page_config(
        page_title="Research Execution Console",
        page_icon=":material/manage_search:",
        layout="wide",
    )
    st.title("Research Execution Console")
    st.caption("Ask a question, inspect the evidence and agent deliverables, and review operational history.")
    use_rag = knowledge_base_controls()
    use_cache = st.sidebar.toggle("Use semantic cache", value=True, key="use_cache")
    model_route = st.sidebar.selectbox(
        "Model route", ["Auto", "Fast", "Quality"], index=0, key="model_route",
    )
    st.sidebar.caption(
        "Auto selects by request complexity. Fast uses a smaller local synthesis "
        "model; Quality uses a stronger local synthesis model. "
        "The Web Searcher always uses qwen2.5:3b."
    )

    if "research_result" not in st.session_state:
        st.session_state.research_result = None
    if "research_execution" not in st.session_state:
        st.session_state.research_execution = None
    st.session_state.setdefault("research_error", None)

    research_tab, analytics_tab = st.tabs(
        ["Research", "Analytics"], key="console_view", on_change="rerun",
    )
    # Rendering either view never executes research. Only form submission does.
    # Keep both tab bodies rendered so input widget state survives navigation.
    with research_tab:
        with st.form("research_form", border=True):
            query = st.text_area(
                "Research query",
                placeholder="What is the Model Context Protocol (MCP)?",
                height=150,
                key="research_query",
            )
            st.caption("Local inference may take some time. Research runs only when you select Research.")
            submitted = st.form_submit_button("Research", type="primary")

        if submitted:
            query = query.strip()
            if not query:
                st.warning("Please enter a research question before starting.")
            else:
                st.session_state.research_result = None
                st.session_state.research_execution = None
                st.session_state.research_error = None
                try:
                    with st.spinner("Running multi-agent research..."):
                        execution = run_research_detailed(
                            query, use_rag=use_rag, use_cache=use_cache,
                            model_route=model_route.lower(),
                        )
                    st.session_state.research_execution = execution
                    if execution.status == "SUCCESS":
                        st.session_state.research_result = execution.final_answer
                except Exception:
                    st.session_state.research_error = (
                        "Research could not be completed. Check that Ollama is running "
                        "and your internet connection is available, then try again."
                    )

        if st.session_state.research_error:
            st.error(st.session_state.research_error)
        if st.session_state.research_execution is not None:
            render_research_console(st.session_state.research_execution)
        else:
            st.caption("Your latest execution summary, evidence, and final answer will appear here.")

    with analytics_tab:
        render_analytics()


if __name__ == "__main__":
    main()
