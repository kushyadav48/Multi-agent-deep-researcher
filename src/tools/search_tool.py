import os

from dotenv import load_dotenv
from linkup import LinkupClient
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


load_dotenv()


class LinkUpSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="The search query to send to LinkUp."
    )
    depth: str = Field(
        default="standard",
        description="Search depth: standard or deep."
    )
    output_type: str = Field(
        default="searchResults",
        description="Output type: searchResults, sourcedAnswer, or structured."
    )


class LinkUpSearchTool(BaseTool):
    name: str = "linkup_search"
    description: str = (
        "Search the web using LinkUp and return relevant web results."
    )
    args_schema: type[BaseModel] = LinkUpSearchInput

    def _run(
        self,
        query: str,
        depth: str = "standard",
        output_type: str = "searchResults",
    ) -> str:

        api_key = os.getenv("LINKUP_API_KEY")

        if not api_key:
            raise ValueError(
                "LINKUP_API_KEY is not set in the environment."
            )

        client = LinkupClient(api_key=api_key)

        response = client.search(
            query=query,
            depth=depth,
            output_type=output_type,
        )

        return str(response)