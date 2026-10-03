from crewai import Crew, Process

from src.tools.search_tool import search_web
from src.tasks.analysis_task import create_analysis_task
from src.tasks.writing_task import create_writing_task


def create_research_crew(query: str) -> Crew:
    # Perform the web search directly in Python.
    # This makes web retrieval deterministic and prevents the local LLM
    # from deciding whether or not to search.
    search_results = search_web(
        query=query,
        max_results=5,
    )

    # Give the raw web evidence directly to the Research Analyst.
    analysis_task = create_analysis_task(
        query=query,
        search_results=search_results,
    )

    # The Technical Writer receives the analyst's output.
    writing_task = create_writing_task(
        query=query,
        analysis_task=analysis_task,
    )

    return Crew(
        agents=[
            analysis_task.agent,
            writing_task.agent,
        ],
        tasks=[
            analysis_task,
            writing_task,
        ],
        process=Process.sequential,
        verbose=True,
    )