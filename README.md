# Multi-Agent Deep Researcher

Research a question using three sequential CrewAI agents:

1. **Web Searcher** uses DDGS for live web research and collects titles, URLs, and descriptions.
2. **Research Analyst** synthesizes web results and relevant local document evidence, preserving attribution and source conflicts.
3. **Technical Writer** produces a Markdown answer with web URLs and document citations.

The project has a Streamlit interface and a separate Model Context Protocol (MCP) stdio server. Both call the same research implementation in `agents.py`.

## Architecture

```text
Streamlit app (app.py)
      |
      v
agents.run_research(query)
      |
      v
Semantic Cache
      +-- HIT --> cached final answer
      |
      +-- MISS
            |
            v
       Local RAG + DDGS
            |
            v
       3 sequential agents
       Web Searcher -> Research Analyst -> Technical Writer
            |
            v
       Final answer -> cache write
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

CrewAI runs exactly three tasks sequentially. RAG is infrastructure, not another agent or task. Retrieval runs before kickoff; the analyst receives bounded local evidence alongside search-task context. The writer receives search and analysis context plus the same bounded local evidence to verify claims and copy exact document citations. Prompts require supported claims, explicit limitations, and source conflicts. The model has a 2,048-token output limit; analysis and writing prompts request at most 350 and 500 words respectively. These prompts do not guarantee factual accuracy or strict word counts; review sources before relying on an answer.

The MCP server runs research in an AnyIO worker thread and serializes calls because stdout redirection is process-wide. CrewAI output goes to stderr, and pending CrewAI event handlers are flushed before stdout is restored to keep the stdio protocol clean.

## Tech stack

- Python 3.11 (validated with 3.11.9)
- CrewAI for agent orchestration and Ollama-compatible model calls
- Ollama running Qwen2.5 3B (`ollama/qwen2.5:3b`)
- DDGS / DuckDuckGo for web research
- MCP Python SDK's FastMCP for the stdio server
- Streamlit for the UI
- AnyIO, Pydantic, and python-dotenv for runtime support
- Qwen3 embeddings (`qwen3-embedding:0.6b`), embedded ChromaDB, and pypdf for local evidence

`requirements.txt` pins the direct dependencies used for validation. FastMCP comes from `mcp.server.fastmcp`; the separate `fastmcp` package is not required. CrewAI supplies the research model client. The Python `ollama` client handles local embeddings.

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

Semantic caching is implemented in Phase 6. Model routing, persistent metrics/observability, analytics dashboards, and formal benchmarking/evaluation remain future work.

## Integrated local knowledge base (Phase 5)

The `rag/` package supports local PDF/TXT/Markdown ingestion, deterministic character chunking, Ollama embeddings, persistent embedded Chroma storage, and retrieval with source/page metadata. It is integrated into the canonical researcher and Streamlit. MCP's existing `crew_research` tool automatically uses the same persisted corpus; no ingestion MCP tool is added.

In Streamlit's **Knowledge Base** sidebar, choose one or more PDF, TXT, or MD files, then click **Add to Knowledge Base**. Choosing files or rerunning the app does not ingest them. Each upload is read from a temporary file that is removed on success or failure; only embedded chunks and metadata persist. **Stored chunks** displays the corpus size. **Use local knowledge base** defaults to enabled; disabling it skips research-time RAG construction, embedding, and retrieval. Corpus status still reads Chroma without embedding, and explicit ingestion remains available.

Start Ollama and install the embedding model in addition to the research model:

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

Defaults: 1,000-character chunks with 200-character overlap, `qwen3-embedding:0.6b` at `http://localhost:11434`, Git-ignored `data/chroma/`, and collection `deep_researcher_documents`. The shared application service anchors storage to the project directory, independent of the launch directory; standalone `RAGService()` retains its working-directory-relative default. PDF pages are numbered from 1 and chunk indices from 0; text files have no page number. Text/Markdown must be UTF-8; PDFs must contain extractable text. There is no OCR or password-input support.

`run_research(query)` remains valid. Optional keyword arguments are `use_rag=True`, `rag_service=None`, `rag_top_k=4`, and `rag_max_distance=0.6`. Service construction is lazy and does not request embeddings. Empty corpora skip query embedding entirely. The researcher retrieves at most four chunks by default and keeps only finite cosine distances at or below the configurable threshold. Cosine distance is `1 - cosine_similarity`; `0.6` requires similarity of at least `0.4`. This is a conservative initial heuristic, not a calibrated relevance guarantee or measured quality improvement. Unknown distances are excluded.

Evidence is rendered deterministically as `LOCAL DOCUMENT EVIDENCE` with numbered blocks containing source filename, page (or `N/A`), chunk index, citation, and content. Each content block is capped at 1,000 characters, and the total local context at 6,000 characters. Embeddings and internal database structures are never included. The analyst synthesizes it with web evidence; the writer preserves citations such as `[Document: protocol_notes.pdf, p. 3]` or `[Document: notes.md]`. Missing pages are omitted, and documents are not represented as web URLs.

If RAG is disabled, empty, or has no relevant chunks, research uses the web path without local evidence. If RAG fails, a warning is logged to stderr and an explicit web-only fallback notice is appended to the returned answer. No retrieval is fabricated; web research remains available. Research failures still use the existing `Error:` response convention.

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

Temporary uploads pass their filename through `ingest_file(path, source=filename)` so reuploading unchanged content from a new temporary directory does not duplicate it. Uploads with the same filename share source identity; changed contents can leave older chunks. A document-management/versioning UI is not implemented.

Model routing, persistent metrics/observability, analytics dashboards, and formal benchmarking/evaluation are not yet implemented. No dependencies were added for Phase 5 or Phase 6.

Validate without Ollama using deterministic fake embeddings and temporary Chroma directories:

```powershell
python -m unittest discover -s tests/rag -t . -v
python -m unittest discover -s tests -t . -v
```

The Phase 5 opt-in smoke uses a temporary corpus and real local embeddings, DDGS, and all three research agents. It checks a unique document fact and its final citation and reports embedding, retrieval, context, and end-to-end timings. Normal unit tests use fakes and mocked research completion, requiring no live Ollama.

```powershell
python -m tests.smoke_phase5
python -m tests.smoke_interfaces  # headless Streamlit HTTP and MCP stdio checks
```

The opt-in live smoke test checks repeated embedding dimensions and verifies that an MCP document ranks above a photosynthesis document. It reports timings and uses a temporary vector store, cleaned after its child process exits:

```powershell
python -m tests.rag.smoke_local
```

## Persistent semantic research cache (Phase 6)

`semantic_cache/` stores final answers in the separate Git-ignored `data/semantic_cache/` directory and collection `deep_researcher_semantic_cache`. The application anchors this path to the project directory. Document chunks remain in `data/chroma/`. Cache records preserve the original query, normalized query, final answer, embedding, Unix creation/expiry timestamps, and typed scope. Chroma uses cosine distance (`1 - similarity`), explicit vectors, and metadata filtering; it never downloads or invokes an automatic embedding model. See [Chroma's collection API](https://docs.trychroma.com/reference/python/collection).

Both RAG and the cache reuse `rag.embeddings.EmbeddingProvider` and `OllamaEmbeddingProvider` with the existing `qwen3-embedding:0.6b` model (observed dimension: 1024). No new embedding implementation, model, database, or dependency is added. Imports and service construction do not request embeddings.

Lookup first checks a deterministic exact-query ID. Normalization strips surrounding whitespace and collapses repeated whitespace while preserving case, punctuation, and identifiers. An exact hit needs no query embedding or similarity search. A non-exact query is embedded only when compatible, unexpired entries exist; the best candidate must have cosine similarity **strictly greater than 0.97** by default and pass a small deterministic intent/polarity/numeric-term guard. A hit immediately returns the stored answer before DDGS, document retrieval, Crew construction/kickoff, or generation calls. RAG-enabled hits still read corpus contents to calculate the current fingerprint.

The configurable default TTL is **3600 seconds (one hour)** because answers use live web sources. Expired records cannot hit. Lowering a service's TTL also limits older entries using their original creation timestamp. Expired records remain on disk; automatic pruning and storage caps are not implemented. An empty or incompatible cache skips query embedding and similarity search. Storing its first answer requires an embedding and may load the local embedding model. On a semantic miss, the lookup vector is reused for insertion. Repeated exact-query/scope writes upsert the same SHA-256 ID.

Scope is a SHA-256 digest over canonical JSON containing cache schema/generation version, research model identity, cache embedding model identity, and RAG mode. RAG scopes additionally include corpus fingerprint, RAG embedding model identity, top-k, maximum retrieval distance, and context/chunk character limits. Web-only mode uses a distinct `WEB_ONLY` namespace. Empty RAG corpora use a deterministic digest of an empty record list. Corpus fingerprints hash sorted stored chunk IDs, actual text, and source/page/chunk metadata, so content edits invalidate cached answers even when IDs/counts stay unchanged. The corpus is checked again before writing a RAG answer; detected changes during research skip that write. Fingerprinting reads the whole corpus and does not add document replacement/versioning or an ingestion transaction. Bump `CACHE_VERSION` when prompts, generation behavior, or safety policy change.

Only successful, nonempty final answers are stored. Exceptions, explicit error responses, incomplete Crew task outputs, and unexpected RAG fallback answers are excluded. Fingerprint, cache lookup, embedding, and cache write failures log warnings and preserve the research path. A cache failure does not turn a successful answer into an error.

Streamlit's **Use semantic cache** toggle defaults to enabled and reaches canonical `run_research()`. MCP retains exactly `crew_research(query)` and uses caching through its existing call; stdio handling and shutdown are unchanged. `run_research(query)` remains valid. Disable caching entirely with `use_cache=False`, which performs no cache construction, scope work, lookup, cache embedding, or write:

```python
from agents import run_research
from semantic_cache.service import SemanticCacheService

answer = run_research("What is the Model Context Protocol?", use_cache=False)
cache = SemanticCacheService(ttl_seconds=1800, similarity_threshold=0.98)
answer = run_research("Explain the Model Context Protocol.", cache_service=cache)
```

### Local threshold calibration

An embedding-only calibration tested four positive pairs, three ordinary negatives, and four hard negatives with the installed model. These are observed similarities, not a formal evaluation:

| Pair | Type | Similarity |
| --- | --- | ---: |
| What is the Model Context Protocol? / Explain the Model Context Protocol. | Positive | 0.987953 |
| How does RAG work? / Explain retrieval augmented generation. | Positive | 0.769329 |
| What is photosynthesis? / Explain photosynthesis. | Positive | 0.910193 |
| How does TCP establish a connection? / Explain how TCP establishes a connection. | Positive | 0.982746 |
| What is MCP? / How does photosynthesis work? | Negative | 0.441671 |
| Model Context Protocol / Transmission Control Protocol definitions | Negative | 0.597673 |
| How does RAG work? / How do I bake sourdough bread? | Negative | 0.364991 |
| Advantages of MCP / disadvantages of MCP | Hard negative | 0.932170 |
| Enable caching in Python / disable caching in Python | Hard negative | 0.923328 |
| TCP connection establishment / termination | Hard negative | 0.833881 |
| MCP definition / security risks of MCP | Hard negative | 0.873958 |

Positive range: **0.769329–0.987953**. Ordinary negative range: **0.364991–0.597673**. Hard-negative range: **0.833881–0.932170**. The **0.97** default exceeds all measured negatives and admits both the MCP and TCP paraphrases. Lower-scoring valid paraphrases deliberately miss. The guard also rejects conflicting advantage/risk, enable/disable, establishment/termination, negation, or numeric terms even if a vector exceeds the threshold. This small sample cannot guarantee precision on arbitrary entities or question intents; semantic caching is heuristic. Raising the threshold or disabling caching is available for stricter reuse. Web freshness is bounded by TTL, not real-time change detection.

### Validation

Normal tests use fake embeddings, mocks, and temporary Chroma directories; they require no live Ollama. Existing Phase 5 researcher tests isolate caching so they continue to exercise their original pipeline assertions.

```powershell
python -m pytest -q tests/semantic_cache tests/rag/test_fingerprint.py tests/test_research_rag.py tests/test_app.py
python -m pytest -q
python -m pip check
python -m tests.smoke_interfaces
```

Explicitly invoked live checks use the existing local model. Calibration and local cache smoke never run CrewAI or DDGS. The research smoke runs exactly two `run_research` calls, first to populate a temporary web-only cache and second to prove semantic reuse. It observes concrete `ddgs.ddgs.DDGS.text`, `Crew.kickoff`, and document-context retrieval, and asserts zero downstream calls on the hit; individual LLM-call counts are not required. The research smoke prints and records its OS temporary cache path and retains storage and Crew logs after success or failure for inspection. The embedding-only local smoke cleans its temporary Chroma storage after child processes release Windows file handles. Do not repeatedly retry a failed research smoke without inspecting the failure.

```powershell
python -m tests.semantic_cache.smoke_calibration
python -m tests.semantic_cache.smoke_local
python -m tests.semantic_cache.smoke_research
```

Formal benchmark/evaluation, model routing, persistent metrics/observability, and analytics dashboards remain unimplemented. No percentage speedup or cost savings are claimed.
