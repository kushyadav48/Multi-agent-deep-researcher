from crewai import Task

from src.agents.writer_agent import create_writer_agent


def create_writing_task(query: str, analysis_task: Task) -> Task:
    writer_agent = create_writer_agent()

    return Task(
        description=(
            f"Write a final research report about: {query}\n\n"
            "Use the research analysis provided as context. "
            "Present the findings clearly and logically. "
            "Organize the report with appropriate headings and sections. "
            "Include important evidence and source information where "
            "available. Do not introduce unsupported claims."
        ),
        expected_output=(
            "A polished and well-structured final research report "
            "containing an introduction, key findings, supporting "
            "evidence, analysis, and a clear conclusion."
        ),
        agent=writer_agent,
        context=[analysis_task],
    )