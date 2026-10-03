import sys
from contextlib import redirect_stdout

from anyio import Lock, to_thread
from crewai.events import crewai_event_bus
from mcp.server.fastmcp import FastMCP

from agents import run_research


mcp = FastMCP("crew_research")
# stdout redirection is process-wide, so research calls must not overlap.
_research_lock = Lock()


@mcp.tool()
async def crew_research(query: str) -> str:
    """Run CrewAI-based research for a query using standard or deep web search.

    Args:
        query: The research query or question.

    Returns:
        The research response from the CrewAI pipeline.
    """
    def run_in_worker() -> str:
        encoding, errors = sys.stderr.encoding, sys.stderr.errors
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        try:
            # MCP already holds its original binary stdout transport.
            with redirect_stdout(sys.stderr):
                try:
                    return run_research(query)
                finally:
                    # Completion handlers can still print after kickoff returns.
                    if not crewai_event_bus.flush(timeout=30.0):
                        raise TimeoutError("CrewAI event handlers did not drain within 30 seconds")
        finally:
            sys.stderr.reconfigure(encoding=encoding, errors=errors)

    async with _research_lock:
        return await to_thread.run_sync(run_in_worker)


if __name__ == "__main__":
    mcp.run(transport="stdio")
