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
        model="ollama/qwen2.5:3b",
        base_url="http://localhost:11434",
        max_tokens=2048,
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
            "relevant information with accurate source links."
        ),
        verbose=True,
        allow_delegation=False,
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
            "and extracting key insights from the provided search results."
        ),
        verbose=True,
        allow_delegation=False,
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
            f"Search the web using DuckDuckGo Search for: {query}. "
            "Return the retrieved titles, exact source URLs, and relevant "
            "descriptions. Do not invent results or URLs. If search fails "
            "or returns no results, report that limitation."
        ),
        agent=web_searcher,
        expected_output=(
            "Detailed raw search results including sources (urls)."
        ),
        tools=[search_tool],
    )

    analysis_task = Task(
        description=(
            f"Analyze the provided search results to answer: {query}. "
            "Use only information supported by those results. Preserve "
            "source URLs exactly and distinguish missing information from "
            "supported facts. Do not invent examples, claims, or citations. "
            "Keep the analysis focused and within 350 words."
        ),
        agent=research_analyst,
        expected_output=(
            "A concise, source-grounded analysis answering the original "
            "question, with exact retrieved source links and any limitations."
        ),
        context=[search_task],
    )

    writing_task = Task(
        description=(
            f"Answer the original question: {query}. "
            "Use the provided search results and analysis to write a "
            "focused markdown answer within 500 words. Cite only exact "
            "URLs present in the search results. Do not invent examples "
            "or unsupported claims. State any search limitations clearly. "
            "Avoid repetition, unrelated topics, and follow-up questions; "
            "finish once the question is answered."
        ),
        agent=technical_writer,
        expected_output=(
            "A concise markdown answer to the original question, grounded "
            "in the retrieved sources, with accurate citations and no repetition."
        ),
        context=[search_task, analysis_task],
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
