from mcp.server.fastmcp import FastMCP

from agents import run_research


mcp = FastMCP("crew_research")


@mcp.tool()
async def crew_research(query: str) -> str:
    """Run CrewAI-based research for a query using standard or deep web search.

    Args:
        query: The research query or question.

    Returns:
        The research response from the CrewAI pipeline.
    """
    return run_research(query)


if __name__ == "__main__":
    mcp.run(transport="stdio")
