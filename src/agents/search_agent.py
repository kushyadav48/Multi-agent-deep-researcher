from crewai import Agent

from src.tools.search_tool import LinkUpSearchTool
from src.config.settings import OLLAMA_MODEL


def create_search_agent() -> Agent:
    search_tool = LinkUpSearchTool()

    return Agent(
        role="Web Search Specialist",
        goal=(
            "Use the web_search tool to research every user query before "
            "producing an answer. Collect accurate, relevant, and up-to-date "
            "information from real web sources. Base your findings only on "
            "information returned by the search tool and preserve the source "
            "URLs in your response."
        ),
        backstory=(
            "You are an expert web researcher. You must perform web searches "
            "instead of relying only on your internal knowledge. You know how "
            "to formulate effective search queries, identify useful sources, "
            "compare information, and collect reliable evidence. Never invent "
            "facts, sources, URLs, people, dates, or statistics. If the search "
            "results do not support a claim, do not present that claim as fact."
        ),
        tools=[search_tool],
        llm=f"ollama/{OLLAMA_MODEL}",
        verbose=True,
        allow_delegation=True,
    )