from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from ddgs import DDGS


def search_web(query: str, max_results: int = 5) -> str:
    """
    Search DuckDuckGo directly and return formatted web results.
    """

    results = DDGS().text(
        query,
        max_results=max_results,
    )

    if not results:
        return "No search results were found."

    formatted_results = []

    for i, result in enumerate(results, start=1):
        title = result.get("title", "No title")
        url = result.get("href", "No URL")
        body = result.get("body", "No description")

        formatted_results.append(
            f"Source {i}:\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Description: {body}\n"
        )

    return "\n".join(formatted_results)


class LinkUpSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="The search query to send to the web search engine."
    )
    depth: str = Field(
        default="standard",
        description="Search depth: standard or deep."
    )
    output_type: str = Field(
        default="searchResults",
        description="Output format."
    )


class LinkUpSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web using DuckDuckGo and return relevant search results."
    )
    args_schema: type[BaseModel] = LinkUpSearchInput

    def _run(
        self,
        query: str,
        depth: str = "standard",
        output_type: str = "searchResults",
    ) -> str:

        max_results = 5 if depth == "standard" else 10

        return search_web(
            query=query,
            max_results=max_results,
        )