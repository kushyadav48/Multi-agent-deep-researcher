from crewai import Agent

from src.config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL


def create_writer_agent() -> Agent:
    return Agent(
        role="Technical Writer",
        goal=(
            "Transform research findings and analysis into a clear, "
            "accurate, well-structured, and easy-to-understand final report."
        ),
        backstory=(
            "You are an expert technical writer who specializes in "
            "turning complex research into concise and well-organized "
            "reports. You present information logically, preserve "
            "important evidence, and make the final report useful "
            "to the reader."
        ),
        llm=f"ollama/{OLLAMA_MODEL}",
        verbose=True,
        allow_delegation=False,
    )