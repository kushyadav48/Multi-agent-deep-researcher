"""Streamlit frontend for the canonical research backend."""

import streamlit as st
from pathlib import Path
from tempfile import TemporaryDirectory

from agents import run_research
from rag.context import get_default_rag_service, source_name


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
        except Exception as error:
            st.warning(f"Knowledge base unavailable: {error}")
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
                except Exception as error:
                    st.error(f"Could not add {name}: {error}")
        try:
            st.caption(f"Stored chunks: {service.count()}")
        except Exception as error:
            st.warning(f"Could not read corpus status: {error}")
    return use_rag


def main():
    st.set_page_config(
        page_title="Multi-Agent Deep Researcher",
        page_icon="🔎",
        layout="centered",
    )
    st.title("🔎 Multi-Agent Deep Researcher")
    st.write(
        "Multiple AI agents search the web with DuckDuckGo, analyze the "
        "information alongside relevant local documents, and synthesize a "
        "research answer with sources."
    )
    st.caption("Local model inference may take some time.")
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

    with st.form("research_form"):
        query = st.text_area(
            "Research query",
            placeholder="What is the Model Context Protocol (MCP)?",
            height=150,
            key="research_query",
        )
        submitted = st.form_submit_button("Research", type="primary")

    if submitted:
        query = query.strip()
        if not query:
            st.warning("Please enter a research question before starting.")
        else:
            st.session_state.research_result = None
            error_message = (
                "Research could not be completed. Check that Ollama is running "
                "and your internet connection is available, then try again."
            )
            try:
                with st.spinner("Researching with multiple agents..."):
                    result = run_research(query, use_rag=use_rag, use_cache=use_cache,
                                          model_route=model_route.lower())
                # The canonical backend can also return errors as strings.
                if result.startswith("Error:"):
                    st.error(error_message)
                else:
                    st.session_state.research_result = result
            except Exception:
                st.error(error_message)

    if st.session_state.research_result is not None:
        st.divider()
        st.subheader("Research Result")
        st.markdown(st.session_state.research_result)


if __name__ == "__main__":
    main()
