from crewai import Agent

from src.tools.search_tool import LinkUpSearchTool
from src.config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL


def create_search_agent() -> Agent:
    search_tool = LinkUpSearchTool()

    return Agent(
        role="Web Search Specialist",
        goal=(
            "Search the web efficiently and find accurate, relevant, "
            "and up-to-date information for the research query."
        ),
        backstory=(
            "You are an expert web researcher. You know how to formulate "
            "effective search queries, identify useful sources, and collect "
            "reliable information from the web."
        ),
        tools=[search_tool],
        llm=f"ollama/{OLLAMA_MODEL}",
        verbose=True,
        allow_delegation=True,
    )