import asyncio
import contextlib
import io
import sys

from fastmcp import FastMCP

from src.services.research_service import run_research


mcp = FastMCP("crew_research")


@mcp.tool()
async def crew_research(query: str) -> str:
    """
    Run the multi-agent research workflow for a given query.
    """
    output = io.StringIO()

    print("SERVER: crew_research tool started", file=sys.stderr)

    with contextlib.redirect_stdout(output):
        result = await asyncio.to_thread(run_research, query)

    print("SERVER: run_research finished", file=sys.stderr)

    return result


@mcp.tool()
async def health_check() -> str:
    """
    Simple MCP health check.
    """
    return "MCP tool execution works"


if __name__ == "__main__":
    mcp.run(transport="stdio")