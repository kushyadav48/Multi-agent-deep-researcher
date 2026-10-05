"""Streamlit frontend for the canonical research backend."""

import streamlit as st

from agents import run_research


def main():
    st.set_page_config(
        page_title="Multi-Agent Deep Researcher",
        page_icon="🔎",
        layout="centered",
    )
    st.title("🔎 Multi-Agent Deep Researcher")
    st.write(
        "Multiple AI agents search the web with DuckDuckGo, analyze the "
        "information, and synthesize a research answer with sources."
    )
    st.caption("Local model inference may take some time.")

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
                    result = run_research(query)
                # The frozen backend can also return errors as strings.
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
