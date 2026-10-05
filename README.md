# Multi-Agent Deep Researcher

Research a question using three sequential CrewAI agents:

1. **Web Searcher** uses DDGS for live web research and collects titles, URLs, and descriptions.
2. **Research Analyst** synthesizes the retrieved evidence and identifies missing information.
3. **Technical Writer** produces a Markdown answer with source links.

The project has a Streamlit interface and a separate Model Context Protocol (MCP) stdio server. Both call the same research implementation in `agents.py`.

## Architecture

```text
Streamlit app (app.py)
      |
      v
agents.run_research(query)
      |
      v
Web Searcher + DDGS
      |
      v
Research Analyst
      |
      v
Technical Writer
      |
      v
Final answer
```

MCP is a separate interface, not an intermediate step for Streamlit:

```text
MCP Client
    |
    v
server.py: crew_research(query)
    |
    v
agents.run_research(query)
```

CrewAI runs the three tasks sequentially. The analyst receives search-task context; the writer receives both search and analysis context. Prompts require exact retrieved URLs, supported claims, and explicit search limitations. The model has a 2,048-token output limit; analysis and writing prompts request at most 350 and 500 words respectively. These prompts do not guarantee factual accuracy or strict word counts; review sources before relying on an answer.

The MCP server runs research in an AnyIO worker thread and serializes calls because stdout redirection is process-wide. CrewAI output goes to stderr, and pending CrewAI event handlers are flushed before stdout is restored to keep the stdio protocol clean.

## Tech stack

- Python 3.11 (validated with 3.11.9)
- CrewAI for agent orchestration and Ollama-compatible model calls
- Ollama running Qwen2.5 3B (`ollama/qwen2.5:3b`)
- DDGS / DuckDuckGo for web research
- MCP Python SDK's FastMCP for the stdio server
- Streamlit for the UI
- AnyIO, Pydantic, and python-dotenv for runtime support

`requirements.txt` pins the direct dependencies to the installed versions used for base-project validation. FastMCP comes from `mcp.server.fastmcp`; the separate `fastmcp` package is not required. CrewAI supplies the model client, so the separate Python `ollama` package is not required either.

## Setup (Windows PowerShell)

From the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python` in the commands below.

Install [Ollama](https://ollama.com/download/windows), start it, and pull the model:

```powershell
ollama pull qwen2.5:3b
```

The backend expects Ollama at `http://localhost:11434`. If it is not already running, start `ollama serve` in another terminal. Internet access is needed for web search. No API keys or environment variables are required for this DDGS + local Ollama setup; no `.env` or `.env.example` is needed. `python-dotenv` remains because `agents.py` uses it to load an optional local environment file.

Run the UI:

```powershell
python -m streamlit run app.py
```

Open `http://localhost:8501`, enter a research query, and select **Research**. The final synthesized answer appears with source links. Local inference can take time. Stop the app with Ctrl+C.

## MCP usage

The server exposes exactly one tool: `crew_research(query: str) -> str`.

```powershell
python server.py
```

This command starts a stdio MCP server and waits for a client's JSON-RPC messages; it is not an interactive query prompt. Configure an MCP client to launch the project's virtual-environment Python and `server.py` using absolute paths. For example, replace `C:\path\to\project` in this configuration:

```json
{
  "mcpServers": {
    "crew_research": {
      "command": "C:\\path\\to\\project\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\project\\server.py"]
    }
  }
}
```

Keep Ollama running while the client calls the research tool.

## Project files

```text
agents.py         Research agents, DDGS tool, tasks, and run_research()
server.py         MCP stdio interface
app.py            Streamlit interface
requirements.txt  Direct runtime dependencies
README.md         Setup and authoritative architecture documentation
.gitignore        Local environment, secrets, and generated-file exclusions
```

## Implementation note

The base workflow follows [the reference project](https://github.com/patchy631/ai-engineering-hub/tree/main/Multi-Agent-deep-researcher-mcp-windows-linux). This implementation intentionally uses DDGS instead of LinkUp and local Qwen2.5 3B instead of the reference's Ollama DeepSeek R1 7B model. It also retains the worker-thread and stdout/event-flush handling needed for the local Windows/CrewAI MCP setup. No search API key is required.

RAG, semantic caching, model routing, metrics, dashboards, and benchmarking extensions are outside this base project.
