from crewai import Task
from src.agents.writer_agent import create_writer_agent


def create_writing_task(query: str, analysis_task: Task) -> Task:
    writer_agent = create_writer_agent()

    return Task(
        description=(
            f"Write a final research report about: {query}\n\n"
            "Use only the verified research analysis provided as context. "
            "Do not add facts from your own internal knowledge.\n\n"
            "Present the verified findings clearly and logically. Organize "
            "the report with appropriate headings and sections.\n\n"
            "Do not include claims that the analyst identified as unsupported, "
            "questionable, contradictory, or unverifiable unless you clearly "
            "describe them as uncertain.\n\n"
            "Never invent names, dates, statistics, quotations, sources, or "
            "URLs. Preserve the real source titles and URLs supplied by the "
            "research analysis.\n\n"
            "Finish the report with a Sources section containing the source "
            "titles and URLs that were actually provided in the research "
            "context. Do not create or guess missing URLs."
        ),
        expected_output=(
            "A polished, evidence-based research report containing an "
            "introduction, verified key findings, supporting evidence, "
            "analysis, a clear conclusion, and a Sources section containing "
            "the real source titles and URLs supplied by the research."
        ),
        agent=writer_agent,
        context=[analysis_task],
    )