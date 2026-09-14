from crewai import Agent

from src.config.settings import OLLAMA_MODEL, OLLAMA_BASE_URL


def create_analyst_agent() -> Agent:
    return Agent(
        role="Research Analyst",
        goal=(
            "Analyze the information collected from web research, "
            "identify the most important and reliable findings, "
            "compare information from different sources, and produce "
            "a clear and well-organized analysis."
        ),
        backstory=(
            "You are an experienced research analyst who specializes "
            "in evaluating information from multiple sources. You "
            "carefully examine evidence, identify contradictions, "
            "separate facts from unsupported claims, and organize "
            "research findings into a logical analysis."
        ),
        llm=f"ollama/{OLLAMA_MODEL}",
        verbose=True,
        allow_delegation=True,
    )