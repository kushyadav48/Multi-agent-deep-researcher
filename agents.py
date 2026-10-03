from typing import Type

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from ddgs import DDGS

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool


load_dotenv()


def get_llm_client():
    """Initialize and return the LLM client."""
    return LLM(
        model="ollama/deepseek-r1:1.5b",
        base_url="http://localhost:11434",
    )


class DuckDuckGoSearchInput(BaseModel):
    """Input schema for DuckDuckGo Search Tool."""

    query: str = Field(
        description="The search query to perform"
    )

    depth: str = Field(
        default="standard",
        description="Depth of search: 'standard' or 'deep'",
    )


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"

    description: str = (
        "Search the web for information using DuckDuckGo "
        "and return relevant results."
    )

    args_schema: Type[BaseModel] = DuckDuckGoSearchInput

    def _run(
        self,
        query: str,
        depth: str = "standard",
    ) -> str:
        """Execute DuckDuckGo search and return results."""

        try:
            max_results = 5 if depth == "standard" else 10

            results = DDGS().text(
                query,
                max_results=max_results,
            )

            if not results:
                return "No search results were found."

            formatted_results = []

            for result in results:
                formatted_results.append(
                    f"Title: {result.get('title', '')}\n"
                    f"URL: {result.get('href', '')}\n"
                    f"Description: {result.get('body', '')}\n"
                )

            return "\n".join(formatted_results)

        except Exception as e:
            return f"Error occurred while searching: {str(e)}"


def create_research_crew(query: str):
    """Create and configure the research crew with all agents and tasks."""

    search_tool = DuckDuckGoSearchTool()

    client = get_llm_client()

    web_searcher = Agent(
        role="Web Searcher",
        goal=(
            "Find the most relevant information on the web, "
            "along with source links (urls)."
        ),
        backstory=(
            "An expert at formulating search queries and retrieving "
            "relevant information. Passes the results to the "
            "'Research Analyst' only."
        ),
        verbose=True,
        allow_delegation=True,
        tools=[search_tool],
        llm=client,
    )

    research_analyst = Agent(
        role="Research Analyst",
        goal=(
            "Analyze and synthesize raw information into structured "
            "insights, along with source links (urls) as citations."
        ),
        backstory=(
            "An expert at analyzing information, identifying patterns, "
            "and extracting key insights. If required, can delegate "
            "fact checking to the 'Web Searcher' only. Passes the final "
            "results to the 'Technical Writer' only."
        ),
        verbose=True,
        allow_delegation=True,
        llm=client,
    )

    technical_writer = Agent(
        role="Technical Writer",
        goal=(
            "Create well-structured, clear, and comprehensive responses "
            "in markdown format, with citations/source links (urls)."
        ),
        backstory=(
            "An expert at communicating complex information "
            "in an accessible way."
        ),
        verbose=True,
        allow_delegation=False,
        llm=client,
    )

    search_task = Task(
        description=(
            f"Search for comprehensive information about: {query}."
        ),
        agent=web_searcher,
        expected_output=(
            "Detailed raw search results including sources (urls)."
        ),
        tools=[search_tool],
    )

    analysis_task = Task(
        description=(
            "Analyze the raw search results, identify key information, "
            "verify facts and prepare a structured analysis."
        ),
        agent=research_analyst,
        expected_output=(
            "A structured analysis of the information with verified facts "
            "and key insights, along with source links."
        ),
        context=[search_task],
    )

    writing_task = Task(
        description=(
            "Create a comprehensive, well-organized response "
            "based on the research analysis."
        ),
        agent=technical_writer,
        expected_output=(
            "A clear, comprehensive response that directly answers "
            "the query with proper citations/source links (urls)."
        ),
        context=[analysis_task],
    )

    crew = Crew(
        agents=[
            web_searcher,
            research_analyst,
            technical_writer,
        ],
        tasks=[
            search_task,
            analysis_task,
            writing_task,
        ],
        verbose=True,
        process=Process.sequential,
    )

    return crew


def run_research(query: str):
    """Run the research process and return results."""

    try:
        crew = create_research_crew(query)

        result = crew.kickoff()

        return result.raw

    except Exception as e:
        return f"Error: {str(e)}"