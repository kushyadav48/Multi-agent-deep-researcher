from src.crew.research_crew import create_research_crew


def run_research(query: str) -> str:
    crew = create_research_crew(query)

    result = crew.kickoff()

    return result.raw