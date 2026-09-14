from crewai import Crew, Process

from src.tasks.search_task import create_search_task
from src.tasks.analysis_task import create_analysis_task
from src.tasks.writing_task import create_writing_task


def create_research_crew(query: str) -> Crew:
    search_task = create_search_task(query)

    analysis_task = create_analysis_task(
        query=query,
        search_task=search_task,
    )

    writing_task = create_writing_task(
        query=query,
        analysis_task=analysis_task,
    )

    return Crew(
        agents=[
            search_task.agent,
            analysis_task.agent,
            writing_task.agent,
        ],
        tasks=[
            search_task,
            analysis_task,
            writing_task,
        ],
        process=Process.sequential,
        verbose=True,
    )