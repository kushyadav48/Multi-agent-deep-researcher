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

`requirements.txt` pins the direct dependencies used for validation. FastMCP comes from `mcp.server.fastmcp`; the separate `fastmcp` package is not required. CrewAI supplies the research model client. The Python `ollama` client is used separately for Phase 4 embeddings.

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

Semantic caching, model routing, metrics, dashboards, and benchmarking extensions remain future work.

## RAG Foundation (Phase 4)

The standalone `rag/` package implements local PDF/TXT/Markdown ingestion, deterministic character chunking, Ollama embeddings, persistent embedded Chroma storage, and similarity retrieval with source/page metadata. It does not change the three-agent researcher: **RAG is not yet integrated into CrewAI or Streamlit**. Document upload UI, semantic cache, and model router are not implemented.

Start Ollama and install only the embedding model needed for this foundation:

```powershell
ollama pull qwen3-embedding:0.6b
```

Use the service from Python:

```python
from rag.service import RAGService

rag = RAGService()
processed_chunks = rag.ingest_file("paper.pdf")  # also .txt and .md
for result in rag.retrieve("What does this document say about MCP?", top_k=5):
    print(result.source, result.page, result.chunk_index, result.distance)
    print(result.text)
```

Defaults: 1,000-character chunks with 200-character overlap, `qwen3-embedding:0.6b` at `http://localhost:11434`, Git-ignored `data/chroma/` (relative to the working directory), and collection `deep_researcher_documents`. PDF pages are numbered from 1 and chunk indices from 0; text files have no page number. Distances use cosine distance; lower is closer. Text/Markdown must be UTF-8; PDFs must contain extractable text. There is no OCR or password-input support.

The `EmbeddingProvider` protocol exposes `embed_text` and `embed_texts` for reuse independently of RAG. Configure it, storage, and chunking explicitly when needed:

```python
from rag.chunking import TextChunker
from rag.embeddings import OllamaEmbeddingProvider
from rag.vector_store import ChromaVectorStore

rag = RAGService(
    OllamaEmbeddingProvider(),
    ChromaVectorStore("data/chroma"),
    TextChunker(chunk_size=1000, overlap=200),
)
```

Stable IDs and upserts prevent duplicate chunks when the unchanged file is ingested repeatedly at the same resolved path with the same chunking settings. Changed/moved files or changed chunk settings can leave older records; automatic replacement and corpus versioning are deferred. Use a separate collection for a different embedding model. `rag.vector_store.clear()` removes this collection's records when a development reset is needed. Ingestion is batched and is not an atomic transaction; errors propagate, and retrying unchanged input safely upserts completed batches. Metadata supports immutable string-keyed scalar values.

Validate without Ollama using deterministic fake embeddings and temporary Chroma directories:

```powershell
python -m unittest discover -s tests/rag -t . -v
python -m unittest discover -s tests -t . -v
```

The opt-in live smoke test checks repeated embedding dimensions and verifies that an MCP document ranks above a photosynthesis document. It reports timings and uses a temporary vector store, cleaned after its child process exits:

```powershell
python -m tests.rag.smoke_local
```
