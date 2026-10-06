# Portfolio: Multi-Agent Deep Researcher

**One-line description:** Built a local-first multi-agent research application that combines web/local evidence, guarded answer reuse, deterministic model selection, and inspectable execution.

**Tech stack:** Python 3.11, CrewAI, MCP/FastMCP, Ollama/Qwen, DDGS, ChromaDB, Streamlit, SQLite, pypdf, Pydantic, AnyIO, python-dotenv, and pytest.

## Engineering features

- Three sequential roles separate web retrieval, analysis, and writing while preserving local citations and actual retrieved web links.
- Local PDF/TXT/Markdown RAG uses explicit embeddings, persistent Chroma, bounded context, stable chunk identity, and source/page metadata.
- Exact/semantic cache reuse respects TTL, model/route identity, corpus state, retrieval settings, and lexical intent guards.
- Deterministic routing avoids an extra model request, keeps Searcher tool use fixed, and exposes scores/reasons and manual overrides.
- Request-scoped traces capture observable deliverables and evidence; a separate SQLite allowlist persists operational metrics without research content.
- Streamlit presents the execution trace and operational analytics. MCP shares the same backend through a protected stdio interface.
- Versioned evaluation separates controlled evidence from live web, bounds expensive work, checkpoints outcomes, and avoids automatic failed-generation retries.

## Measured results

All results are from the unchanged [Phase 10 report](benchmarks/phase10_baseline.md) and [structured baseline](../evaluation/results/phase10_baseline.json).

| Result | Defensible scope |
| --- | --- |
| 100% router policy-conformance | 36 curated queries: 18 expected FAST and 18 expected QUALITY. |
| 100% cache precision, 40% recall, 70% accuracy | 20 curated pairs; TP=4, TN=10, FP=0, FN=6. |
| 100% Hit@1/Hit@4 and MRR 1.0 | Controlled six-document synthetic corpus and ten retrieval queries; mean retrieval 50.32 ms. |
| 32.91 s → 32.98 ms, 997.95× speedup | One measured semantic-cache paraphrase pair, web-only scope, 99.90% latency reduction. |
| FAST 36.47 s / 6,477.5 tokens; QUALITY 28.26 s / 4,449.5 tokens | Mean total latency/reported tokens on two controlled prompts per route; full content-presence rubric coverage on both. |
| Local fact propagation | One controlled integration request retrieved Meridian's synthetic `Cedar-47` fact and preserved local attribution. |

QUALITY was faster and used fewer reported tokens in the tiny controlled sample. That finding does not establish a general route ranking; FAST describes the smaller synthesis model. The cache speedup applies to one pair, and retrieval metrics do not establish universal answer quality.

## Potential resume bullets

- Built a local-first multi-agent research platform using CrewAI, MCP, Ollama, Streamlit, Chroma, and DDGS, integrating local RAG, semantic caching, deterministic routing, structured execution traces, and SQLite analytics.
- Reduced repeated-query latency from 32.91 s to 32.98 ms (997.95×) on one measured semantic-cache paraphrase pair; measured 100% precision and 40% recall on a curated 20-pair cache evaluation.
- Evaluated routing with 100% policy-conformance on 36 curated queries and RAG retrieval with 100% Hit@1 and MRR 1.0 on a controlled ten-query synthetic corpus, supported by 215 passing automated tests and 20 passing subtests in final Phase 11 validation.

## Interview talking points

| Question | Answer |
| --- | --- |
| Why three agents? | The roles separate evidence collection, synthesis, and writing, giving inspectable task deliverables and explicit sequential context. This is a design choice; no benchmark proves three agents beat one. |
| Why deterministic routing? | String/regex rules select without another model call and expose reproducible scores/reasons. The trade-off is limited lexical understanding; manual overrides address mismatches. |
| Why keep Searcher on Qwen2.5 3B? | It was validated for CrewAI/DDGS tool calling. Changing only Analyst/Writer isolates the route's synthesis model while retaining known tool behavior. |
| Why strict similarity >0.97? | Reusing the wrong answer can be worse than a miss, so the cache favors precision with scope and intent guards. Phase 10 measured no false positives but six missed valid paraphrases: 40% recall. |
| Why separate fixed evidence from live web? | Fixed evidence controls source variation for route/cache/RAG experiments; live web checks the external boundary separately. Mixing them would obscure what caused a result. |
| Why SQLite? | Standard-library embedded storage is enough for local operational history, with parameterized inserts, an explicit content allowlist, and no additional database service. Aggregation/retention are still limited. |
| Why Chroma? | It persists explicit local embeddings with cosine retrieval and metadata. Separate document/cache collections prevent confusing evidence with generated answers. |
| Why request-scoped observation? | Each request owns its tool and recorder, preventing concurrent evidence mixing. Rich content stays in the current result while SQLite stores only operational metadata. |
| What unexpected result did evaluation reveal? | QUALITY was faster and lower-token than FAST on two controlled prompts per route. Model size alone did not predict request latency; model warmth, output length, tool calls, and workload matter. The sample remains too small for a general conclusion. |
| Does citation preservation ensure correctness? | It keeps actual retrieved links available and prompts exact local attribution, but it does not verify every claim or citation entailment. Source review remains necessary. |

## Limits and lessons learned

The router is heuristic, cache intent guards are incomplete, and the curated cache set showed low recall. Synthetic retrieval and substring rubric coverage are narrow evaluation signals. No OCR or automatic document replacement/version UI exists; changed uploads can leave older chunks. Rich traces are session-scoped, metrics have no retention policy, per-agent timing is absent, and token reporting depends on the provider. Local hardware/model warmth affect latency, and DDGS queries leave the machine.

The central lesson is to measure implementation behavior rather than equate a smaller model with a faster request or a high-similarity vector with identical intent. Benchmarks should state corpus, sample size, measurement boundary, and failures beside their successes. See [architecture](architecture.md) for decisions and [demo](demo.md) for a presentation script and real-screenshot checklist.

## Project evolution

Base Crew/MCP workflow → local RAG → semantic cache → deterministic model routing → structured traces/SQLite metrics → Research Execution Console/Analytics → reproducible evaluation → final portfolio documentation. The production backend and dependency pins remain frozen during the final documentation phase.
