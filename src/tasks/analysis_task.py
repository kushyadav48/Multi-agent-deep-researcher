from crewai import Task

from src.agents.analyst_agent import create_analyst_agent


def create_analysis_task(query: str, search_task: Task) -> Task:
    analyst_agent = create_analyst_agent()

    return Task(
        description=(
            f"Analyze the web research collected for the following topic: {query}\n\n"
            "Review the search findings carefully. Identify the most "
            "important facts, compare information from different sources, "
            "identify contradictions or unsupported claims, and determine "
            "which information is most reliable."
        ),
        expected_output=(
            "A detailed research analysis containing the key findings, "
            "important evidence, source comparisons, potential "
            "contradictions, and a clear conclusion based on the "
            "available research."
        ),
        agent=analyst_agent,
        context=[search_task],
    )
