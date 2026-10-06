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
agents.run_research_detailed(query)
      |
      v
Deterministic complexity router (Auto / Fast / Quality)
      |
      v
Semantic Cache
      +-- HIT --> cached answer + bypass trace -------+
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
       Final answer -> cache write                  |
            |                                       |
            v                                       v
       ResearchExecutionResult <--------------------+
       Routing / Cache / RAG / Web / Agent outputs / Timings / Final answer
            |
            v
       SQLite metrics (operational metadata only)
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
- Ollama: fixed Qwen2.5 3B search; Qwen3 1.7B or Qwen2.5 3B synthesis
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

Install [Ollama](https://ollama.com/download/windows), start it, and pull the generation models:

```powershell
ollama pull qwen2.5:3b
ollama pull qwen3:1.7b
```

The backend expects Ollama at `http://localhost:11434`. If it is not already running, start `ollama serve` in another terminal. Internet access is needed for web search. No API keys or environment variables are required for this DDGS + local Ollama setup; no `.env` or `.env.example` is needed. `python-dotenv` remains because `agents.py` uses it to load an optional local environment file.

Run the UI:

```powershell
python -m streamlit run app.py
```

Open `http://localhost:8501`, enter a research query, and select **Research**. The Research Execution Console shows the final synthesized answer and the observable execution trace. Local inference can take time. Stop the app with Ctrl+C.

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
observability/    Typed execution results, request recorder, SQLite metrics
server.py         MCP stdio interface
app.py            Streamlit entry point, research form, knowledge-base controls
ui/               Research console, operational analytics, pure formatting helpers
requirements.txt  Direct runtime dependencies
README.md         Setup and authoritative architecture documentation
.gitignore        Local environment, secrets, and generated-file exclusions
```

## Implementation note

The base workflow follows [the reference project](https://github.com/patchy631/ai-engineering-hub/tree/main/Multi-Agent-deep-researcher-mcp-windows-linux). This implementation intentionally uses DDGS instead of LinkUp and local Qwen2.5 3B instead of the reference's Ollama DeepSeek R1 7B model. It also retains the worker-thread and stdout/event-flush handling needed for the local Windows/CrewAI MCP setup. No search API key is required.

Semantic caching is implemented in Phase 6, deterministic local model routing in Phase 7, structured observability with persistent operational metrics in Phase 8, and the Research Execution Console with Analytics in Phase 9. Formal benchmarking/evaluation (Phase 10) remains future work.

## Research Execution Console (Phase 9)

The single Streamlit app has **Research** and **Analytics** tabs. The sidebar preserves explicit PDF/TXT/MD ingestion, stored chunk counts, the local knowledge-base toggle, semantic-cache toggle, and Auto/Fast/Quality routing. Upload selection never ingests automatically. Research runs only on form submission through `run_research_detailed()`; tab navigation, analytics refresh, and ordinary reruns do not repeat research. The latest structured result stays in that browser session until another request, tab closure, or server restart.

The Research view displays the submitted question and a visible **Execution Summary**: SIMPLE/COMPLEX classification, FAST/QUALITY route, cache decision, RAG/web usage, chunk/result counts, total runtime, and reported tokens. Searcher, Analyst, and Writer model assignments come directly from the captured routing trace, including manually selected routes.

Collapsed sections provide the observable execution details:

- **Routing Decision:** requested mode, selected route, classification, score, quality threshold, policy version, and human-readable reasons.
- **Semantic Cache:** status, available similarity/distance, configured strict threshold, and reason. Exact and semantic hits clearly indicate that Crew, DDGS, and RAG were not executed.
- **RAG Evidence:** status, retrieval time, chunk count, source, actual page when present, chunk index, and available distance. Full retrieved text lives in individual expanders; nonpaged documents have no invented page.
- **Web Search Results:** each actual DDGS call has its own query, duration, status, and returned results. Titles, HTTP(S) source links, and snippets come from the trace; rendering performs no search.
- **Web Searcher Output**, **Research Analyst Output**, and **Technical Writer Output:** task status, actual model, and captured observable task deliverable, or a clear unavailable/bypass reason. Raw Writer output remains distinct from the final answer.
- **Execution Timings:** routing, cache lookup, RAG retrieval, web search, Crew, and total. Values below one second use milliseconds; longer values use seconds. Unexecuted stages say “Not executed.” Web time is included in Crew time, so stage timings are not additive. Input/output/total token counts say “Not reported” when absent; no token counts or costs are estimated.
- **Request Details:** request ID, UTC timestamps, and metrics persistence status.

**Final Research Answer** is a prominent, always-open Markdown section. Warnings remain visible. Failed requests retain their observable trace and show a concise error category rather than raw exception text or a Python traceback. Intermediate agent content is observable task output, not private reasoning: hidden chain-of-thought, scratchpads, prompts, provider messages, credentials, and embeddings are never rendered.

The Analytics view reads `MetricsStore.summary()` and `recent(limit=20)` from `data/metrics/research_metrics.db`. It shows total/successful/failed requests, cache hit rate and hits/eligible lookups, FAST/QUALITY usage, average/P50/P95 latency, RAG/DDGS request usage, total DDGS calls, and a recent-request table with status, route, cache status, RAG use, web counts, runtime, and tokens. Hit rate excludes disabled, bypassed, and failed cache lookups. Summary cards cover all history; the recent-request table covers the latest 20 requests and includes measured latency in its Total Time column. Missing latency is explicitly unavailable. **Refresh Analytics** reloads history without polling or running research.

Persistent analytics store no raw queries, final answers, agent outputs, RAG text, or web snippets. The table displays only a short query-hash prefix and operational fields, allowlisted before being sent to the browser. Rich research content is rendered only from the current session result. An empty database shows a clear empty state, and an unreadable metrics store shows a warning while Research remains usable.

The `ui/` package keeps rendering separate from execution and pure formatting. Analytics uses native Streamlit Markdown tables, avoiding dataframe/chart DLL loading on systems where Windows Application Control blocks the installed pandas binaries. Deterministic trace fixtures and Streamlit `AppTest` cover the console, session reruns, cache bypasses, knowledge-base regression, and analytics against temporary SQLite stores without live Ollama or DDGS calls. No dependencies or backend policies change in Phase 9.

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

Structured observability and operational metrics are described under Phase 8 below. The analytics dashboard and formal benchmarking/evaluation are not yet implemented. No dependencies were added for Phases 5 through 8.

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

Scope is a SHA-256 digest over canonical JSON containing cache schema/generation version, router policy version, selected route, fixed search model, synthesis model identity, cache embedding model identity, and RAG mode. RAG scopes additionally include corpus fingerprint, RAG embedding model identity, top-k, maximum retrieval distance, and context/chunk character limits. Web-only mode uses a distinct `WEB_ONLY` namespace. Empty RAG corpora use a deterministic digest of an empty record list. Corpus fingerprints hash sorted stored chunk IDs, actual text, and source/page/chunk metadata, so content edits invalidate cached answers even when IDs/counts stay unchanged. The corpus is checked again before writing a RAG answer; detected changes during research skip that write. Fingerprinting reads the whole corpus and does not add document replacement/versioning or an ingestion transaction. Bump `CACHE_VERSION` when prompts, generation behavior, or safety policy change.

Only successful, nonempty final answers are stored. Exceptions, explicit error responses, incomplete Crew task outputs, and unexpected RAG fallback answers are excluded. Fingerprint, cache lookup, embedding, and cache write failures log warnings and preserve the research path. A cache failure does not turn a successful answer into an error.

Streamlit's **Use semantic cache** toggle defaults to enabled and reaches the canonical detailed pipeline. MCP retains exactly `crew_research(query)` and uses caching through its existing string wrapper; stdio handling and shutdown are unchanged. `run_research(query)` remains valid. Disable caching entirely with `use_cache=False`, which performs no cache construction, scope work, lookup, cache embedding, or write:

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

Formal benchmark/evaluation (Phase 10) and analytics dashboards remain unimplemented. Phase 8 adds operational observability below. No percentage speedup or cost savings are claimed.

## Deterministic local model routing (Phase 7)

`run_research(query)` defaults to `model_route="auto"`. Routing in `routing/` uses only Python string/regex rules: **zero LLM calls and zero embedding calls**. It selects once, before semantic-cache scope and lookup, so cache hits still bypass document retrieval, Crew construction/kickoff, DDGS, and generation.

| Selected route | Web Searcher (fixed) | Research Analyst | Technical Writer |
| --- | --- | --- | --- |
| Fast | `ollama/qwen2.5:3b` | `ollama/qwen3:1.7b` | `ollama/qwen3:1.7b` |
| Quality | `ollama/qwen2.5:3b` | `ollama/qwen2.5:3b` | `ollama/qwen2.5:3b` |

Exactly three agents, sequential task order, DDGS tool access, citation instructions, 2,048-token generation limits, and all RAG settings/evidence remain the same. Embeddings remain `qwen3-embedding:0.6b`. Only the Analyst/Writer model is selected. Streamlit's **Model route** control defaults to **Auto**; **Fast** forces the smaller synthesis model and **Quality** forces the stronger synthesis model. MCP keeps its existing `crew_research(query)` API and uses Auto.

Policy **v1** adds these scores, with each category counted at most once:

| Complexity signal | Score |
| --- | ---: |
| At least 30 whitespace-delimited words | +1 |
| At least 80 words | +1 additional |
| Comparison | +2 |
| Trade-offs or pros/cons | +2 |
| Evaluation, critique, risk analysis, contradictions, or evidence quality | +3 |
| Recommendation or architecture/system design | +3 |
| Multiple sources/documents | +1 |
| At least two action words joined by “and”, comma, semicolon, or newline | +1 |

Auto chooses **Quality at score >= 3**, otherwise **Fast**. “What is MCP?”, “Explain RAG.”, and “Summarize MCP in three bullets.” use Fast. “Critique this proposal.” and a request to compare MCP/REST, analyze security trade-offs, and recommend enterprise architecture use Quality. `RoutingDecision` records the requested mode, selected route, models, score, deterministic reasons, and policy version. This small lexical heuristic can misread unusual wording or non-English requests; manual overrides are available.

```python
from agents import run_research
from routing import route_query

decision = route_query("What is MCP?")  # Fast; no inference
answer = run_research("Explain RAG.", model_route="fast")
answer = run_research("Summarize MCP.", model_route="quality")
```

Cache generation version is **2**. Route, search/synthesis models, and router policy participate in scope alongside the existing corpus/retrieval identity. Fast and Quality cannot cross-hit. Auto and a manual override selecting the same route can reuse the same compatible answer; requested mode, score, and reasons are not part of scope. Phase 6 records naturally miss without deleting databases or old entries. TTL remains 3600 seconds, and semantic similarity must remain **strictly > 0.97**. The selected decision is reused when rechecking scope before writing.

There is no runtime quality judge, automatic Fast-to-Quality escalation, or second generation pass. No Python dependency changes or measured speedup claims are introduced.

If synthesis omits retrieved web links, the backend appends a concise **Sources retrieved** list using only exact URLs captured by that request's DDGS tool. Existing links are not duplicated, error/empty answers are not expanded, and no additional search or generation occurs. This preserves access to the evidence; it does not verify every generated claim or replace inline attribution. The returned answer, including these links, is what the semantic cache stores.

Offline validation and the explicitly invoked, bounded single-request regression:

```powershell
python -m pytest -q tests/routing tests/test_research_rag.py tests/semantic_cache tests/test_app.py
python -m pytest -q
python -m tests.routing.smoke_research
python -m tests.smoke_interfaces
```

The routing regression disables cache and RAG, runs one simple Auto/Fast research request, observes concrete `crewai.crew.Crew.kickoff` and `ddgs.ddgs.DDGS.text`, and records actual agent assignments, all three task outputs, retrieved/cited URLs, and runtime in Git-ignored `validation_logs/`. Quality is verified through static Crew construction without kickoff. It is a smoke check, not a formal performance benchmark.

## Structured execution trace and persistent metrics (Phase 8)

`observability/` separates the rich **current in-memory execution trace** from the small **persistent operational history**. There is one research implementation: `run_research()` delegates to `run_research_detailed()` and returns `result.final_answer` as a string, including the existing `Error:` convention. MCP keeps exactly its existing tool and protocol. Streamlit uses the detailed API, retains the current result in `st.session_state.research_execution`, and continues displaying only the final answer with the existing upload/RAG/cache/routing controls. **Phase 9 will render the Research Execution Console; it is not implemented here.**

```python
from agents import run_research_detailed
from observability import MetricsStore

result = run_research_detailed(
    "What is MCP?", use_rag=True, rag_service=None,
    rag_top_k=4, rag_max_distance=0.6,
    use_cache=True, cache_service=None, model_route="auto",
    metrics_store=None, record_metrics=True,
)
print(result.final_answer)
trace = result.to_dict()  # JSON-ready, including derived stage counters

store = MetricsStore()  # same project-anchored default database
recent = store.recent(limit=20)
summary = store.summary()
count = store.count()
```

`ResearchExecutionResult` contains a UUID `request_id`, query, UTC ISO start/finish timestamps, success/error status, routing/cache/RAG/web traces, three agent records, timings, nullable usage, warnings/error types, final answer, and `metrics_persisted`. The trace captures only observable outputs:

- Routing reuses the exact execution decision: requested mode, selected route, SIMPLE (Fast) / COMPLEX (Quality), score, threshold, reasons, policy version, and all three model identities. Classification follows the selected route, including manual overrides.
- Cache status is `DISABLED`, `EMPTY` (no live compatible entries, inferred from the existing lookup), `MISS`, `EXACT_HIT`, or `SEMANTIC_HIT`. `ERROR` reports cache failure; `BYPASSED` means lookup did not execute. Available hit fields include similarity, distance (`1 - similarity`), entry ID, and configured threshold. No additional lookup occurs.
- RAG reports enabled/used, retrieved relevant chunk count, duration, and status: `DISABLED`, `EMPTY_CORPUS`, `NO_RELEVANT_RESULTS`, `USED`, `ERROR` (existing web-only fallback), `BYPASSED` (cache hit), or `NOT_EXECUTED`. Evidence contains filename, nullable page, chunk index, distance, and retrieved text. This is the relevant retrieval result; existing context limits still bound what reaches the agents. Non-paged documents keep `page=None`. No embedding or arbitrary document metadata is copied.
- Each real DDGS tool invocation records its actual query, returned title/URL/snippet data, duration, and `SUCCESS`/`EMPTY`/`ERROR` status. Calls and result counts include repeated invocations; error calls include only the error type. The concrete request-local tool retains its own calls, including through CrewAI's adapter. There is no global last-result state, second search, production monkey-patch, or scraping of generated URLs.
- Agent records follow the known Searcher/Analyst/Writer task order, with task/agent name, model, status, output text, and skip reason. Only public `TaskOutput.raw` is copied. Cache hits use `NOT_EXECUTED`, `output_text=None`, and `reason="semantic_cache_hit"`; RAG and DDGS are skipped. Missing task outputs are marked `UNAVAILABLE`; unexpected output counts add a warning. A failed Crew retains available completed public task outputs and marks remaining tasks as errors.
- The Writer's returned task output remains separate from `final_answer`, which may append missing retrieved web links or the existing RAG fallback notice.

Elapsed milliseconds use `time.perf_counter()`: `routing_ms`, `cache_lookup_ms` (including scope/setup), `rag_retrieval_ms`, `web_search_ms` (sum of actual DDGS calls), `crew_ms`, and `total_ms`. **None means not executed**, including Crew/RAG/web on a cache hit. Web time is included in Crew time, so stage values should not be summed to calculate total. Total covers research, setup, and cache writes, excluding metrics persistence. Agent-specific timings are not collected.

The installed CrewAI public `token_usage` provides prompt/completion/total counters. Input/output/total tokens are copied only when nonnegative, internally consistent, and total is positive; absent, default-zero, or inconsistent counters remain `None`. Local inference providers may not report usage reliably. There are no invented counts or API-dollar cost calculations.

Hidden chain-of-thought, private scratchpads, provider reasoning, internal prompts/messages, credentials, and raw embeddings are never collected by this subsystem. Rich outputs remain in the current execution result only. Existing semantic-cache and RAG persistence retain their preexisting content behavior; the metrics database has its own stricter content boundary.

### SQLite operational history

Default storage is Git-ignored **`data/metrics/research_metrics.db`**, anchored to the project root. Python's standard-library `sqlite3` uses schema version **1** (`PRAGMA user_version`), a `requests` table keyed by request ID, a timestamp index, WAL mode, a five-second busy timeout, parameterized values, and short-lived connections. No new dependency or database server is required. Unknown schema versions are rejected without overwriting them.

Each success, cache hit, or failed request writes one row containing request ID, UTC timestamps, status, SHA-256 query hash and character count, requested/selected route and score, three model identities, cache status/similarity, RAG enabled/used/count, DDGS call/web-result counts, the six timing fields, nullable input/output/total tokens, and error type. SHA-256 hashes the **exact UTF-8 query**, without modifying semantic-cache normalization. Hashes identify repeated exact queries; they are deterministic digests, not encryption.

An explicit column allowlist excludes raw query text, agent outputs, final answers, RAG text, web queries/snippets/URLs, prompts, credentials, and embeddings. SQLite errors never print to MCP stdout. A metrics initialization or write failure preserves the research answer, leaves `metrics_persisted=False`, and adds a warning containing only the exception type. `record_metrics=False` disables metrics construction and writes; `metrics_store=MetricsStore(custom_path)` supports isolated histories and tests.

`count()` returns stored request count. `recent(limit=20)` returns operational rows newest first; limits must be positive integers. `summary()` returns total/success/failed requests, cache hits/lookups and hit rate, Fast/Quality counts, average/p50/p95 latency, requests using DDGS, total DDGS calls, and requests using RAG. Cache hit rate divides hits by successful enabled lookups (`EMPTY`, `MISS`, and both hit statuses), excluding disabled/bypassed/error lookups. Empty denominators and empty latency sets return `None`. Percentiles use linear interpolation at `(n - 1) * percentile`; latency aggregates include successful and failed requests with measured totals.

History has no retention policy or full trace archive yet. Summary percentile calculation loads operational rows into Python; it is intended for lightweight local history. No charts, cloud telemetry, cost tracking, router/cache-policy changes, evaluation harness, or extra agent are added.

### Validation

Deterministic tests use fake Crew completions/DDGS results, fake embeddings, and temporary databases. Default metrics are isolated per test, preventing pollution of application history.

```powershell
python -m pytest -q tests/observability tests/test_research_rag.py tests/semantic_cache tests/routing tests/rag tests/test_app.py
python -m pytest -q
python -m pip check
python -m tests.smoke_interfaces
```

The opt-in observability regression runs **exactly one** live research request with cache/RAG disabled and Auto routing, using a temporary metrics DB and a 20-minute child-process bound. It independently observes concrete `crewai.crew.Crew.kickoff` and `ddgs.ddgs.DDGS.text`, compares real queries/results/counts with the structured trace, checks all three public task outputs and the separate final answer, validates timings and the single operational row, and confirms raw query absence from SQLite. It never instruments `agents.LLM.call`. Selected observable evidence is checkpointed to ignored `validation_logs/phase8_research_regression.json`; verbose console rendering is discarded. Failures do not trigger another generation. Normal interface smoke initializes MCP/lists its one tool/disconnects and checks headless Streamlit HTTP; it performs no research.

```powershell
python -m tests.observability.smoke_research
```
