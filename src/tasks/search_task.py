from crewai import Task

from src.agents.search_agent import create_search_agent
from src.tools.search_tool import search_web


def create_search_task(query: str) -> Task:
    search_agent = create_search_agent()

    # Perform the web search directly in Python.
    # This guarantees that real search results are available to the agent,
    # even if the local LLM does not call the CrewAI web_search tool.
    search_results = search_web(
        query=query,
        max_results=5,
    )

    return Task(
        description=(
            f"Research the following topic: {query}\n\n"

            "A web search has already been performed for you by the Python "
            "application. The real search results are provided below.\n\n"

            "================ WEB SEARCH RESULTS ================\n"
            f"{search_results}\n"
            "====================================================\n\n"

            "Use ONLY the supplied web search results as factual evidence. "
            "Do not answer from your internal knowledge.\n\n"

            "Extract the information that directly answers the research "
            "question. Preserve the source titles and URLs exactly as they "
            "appear in the supplied results.\n\n"

            "Do not invent names, dates, statistics, quotations, facts, "
            "sources, or URLs. If the supplied search results do not contain "
            "enough evidence to answer something, explicitly state that it "
            "could not be verified from the available search results.\n\n"

            "For every factual finding that you produce, you MUST include "
            "the exact Source Title and Source URL from the supplied web "
            "search results immediately below that finding."
        ),
        expected_output=(
            "A structured evidence-based research result using EXACTLY this "
            "format:\n\n"

            "FINDINGS:\n"

            "- Finding: <verified factual finding>\n"
            "  Source Title: <exact title from supplied search results>\n"
            "  Source URL: <exact URL from supplied search results>\n\n"

            "Repeat the above structure for each important finding.\n\n"

            "Every finding MUST contain a Source Title and Source URL copied "
            "exactly from the supplied web search results. Do not output a "
            "finding without its supporting source. Do not invent or modify "
            "URLs."
        ),
        agent=search_agent,
    )