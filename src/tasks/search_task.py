from crewai import Task

from src.agents.search_agent import create_search_agent


def create_search_task(query: str) -> Task:
    search_agent = create_search_agent()

    return Task(
        description=(
            f"Research the following topic: {query}\n\n"
            "Search the web for accurate, relevant, and up-to-date "
            "information. Use reliable sources and collect the key "
            "facts and evidence needed for further analysis."
        ),
        expected_output=(
            "A collection of relevant web research findings, including "
            "important facts, source information, and evidence related "
            "to the research topic."
        ),
        agent=search_agent,
    )