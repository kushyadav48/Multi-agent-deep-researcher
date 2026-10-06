# Phase 10 baseline (phase10-v1)

## 1. Benchmark Environment

- Timestamp (UTC): 2026-10-06T20:09:12.054229+00:00
- Production checkpoint commit: `d32671f5a16ac0a7e02c1bf4c019ffab9942696d`
- Harness/dataset SHA-256: `86e6f066b75fe0eea1d43e6c90b5754c3960c6813273d00795bea973ee3d720e`
- Harness files were evaluated before their Phase 10 commit; the hash identifies that exact methodology.
- Python: 3.11.9; platform: Windows-10-10.0.26300-SP0
- Research invocations started: 8; research reservations: 8; observed Crew executions: 7; Crew reservations: 7.

| Configuration | Value |
| --- | --- |
| cache_similarity_threshold | 0.9700 |
| cache_ttl_seconds | 3600.0000 |
| cache_version | 2 |
| embedding_model | qwen3-embedding:0.6b |
| fast_synthesis | ollama/qwen3:1.7b |
| max_crew_executions | 7 |
| max_research_invocations | 8 |
| quality_synthesis | ollama/qwen2.5:3b |
| quality_threshold | 3 |
| rag_max_distance | 0.6000 |
| rag_top_k | 4 |
| router_policy | v1 |
| searcher | ollama/qwen2.5:3b |

## 2. Methodology

OFFLINE uses the real deterministic router, production cache lookup, and production RAG retrieval. Cache and RAG use real local embeddings, but no generation or DDGS network.
CONTROLLED EVIDENCE uses the canonical detailed research API and patches only `ddgs.ddgs.DDGS.text`. Both routes receive identical fixed evidence for each of two prompts. Cache reuse and local evidence propagation are separate cases.
LIVE WEB has one research case and permits at most one real DDGS boundary call. Any additional tool attempt is blocked and reported. No failed case is retried automatically.
All Chroma/cache/metrics stores are isolated under the ignored benchmark output directory. Atomic checkpoints retain completed cases. Interrupted reservations are never automatically retried. No generated answers, agent outputs, prompts, vectors, credentials, or verbose logs appear in this report or result JSON.
Durations are milliseconds from the production trace (or `perf_counter` for offline cases). Research trace timings exclude harness setup, semantic-hit preflight, event draining, and metrics persistence. Python process CPU time excludes the Ollama service and GPU. It is coordinator work, not total inference resource usage. Small synthesis groups report N, mean, and median; no p95 is claimed. Each prompt/route has one sample, in FAST then QUALITY order, without generation warm-up or a fixed random seed.
The frozen MCP evidence paraphrases the [official architecture documentation](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture). Hybrid evidence paraphrases [Microsoft's hybrid-search documentation](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview); its deployment trade-off recommendation is explicitly a benchmark engineering inference. Synthetic project facts are original benchmark content.

| Local model | Digest | Bytes | Quantization |
| --- | --- | --- | --- |
| qwen3:1.7b | 8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7 | 1359293444 | Q4_K_M |
| qwen3-embedding:0.6b | ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d | 639150858 | Q8_0 |
| qwen2.5:3b | 357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b | 1929912432 | Q4_K_M |

## 3. Router Policy Evaluation

Category: **OFFLINE**. Attempted cases: 36.

Curated policy-conformance accuracy: 1.0000 (36/36). This set checks v1 behavior, including bare comparisons that remain FAST and lexical false alarms such as 'recommendation engine'; it is not general routing intelligence.

| Expected / Actual | FAST | QUALITY |
| --- | --- | --- |
| FAST | 18 | 0 |
| QUALITY | 0 | 18 |

| Case | Expected | Actual | Score | Status |
| --- | --- | --- | --- | --- |
| route-01 | FAST | FAST | 0 | PASS |
| route-02 | FAST | FAST | 0 | PASS |
| route-03 | FAST | FAST | 0 | PASS |
| route-04 | FAST | FAST | 0 | PASS |
| route-05 | FAST | FAST | 0 | PASS |
| route-06 | FAST | FAST | 0 | PASS |
| route-07 | FAST | FAST | 0 | PASS |
| route-08 | QUALITY | QUALITY | 3 | PASS |
| route-09 | FAST | FAST | 2 | PASS |
| route-10 | FAST | FAST | 2 | PASS |
| route-11 | QUALITY | QUALITY | 4 | PASS |
| route-12 | QUALITY | QUALITY | 3 | PASS |
| route-13 | QUALITY | QUALITY | 3 | PASS |
| route-14 | QUALITY | QUALITY | 3 | PASS |
| route-15 | QUALITY | QUALITY | 3 | PASS |
| route-16 | QUALITY | QUALITY | 3 | PASS |
| route-17 | QUALITY | QUALITY | 3 | PASS |
| route-18 | QUALITY | QUALITY | 3 | PASS |
| route-19 | QUALITY | QUALITY | 3 | PASS |
| route-20 | FAST | FAST | 1 | PASS |
| route-21 | FAST | FAST | 1 | PASS |
| route-22 | QUALITY | QUALITY | 3 | PASS |
| route-23 | QUALITY | QUALITY | 3 | PASS |
| route-24 | QUALITY | QUALITY | 7 | PASS |
| route-25 | QUALITY | QUALITY | 3 | PASS |
| route-26 | FAST | FAST | 0 | PASS |
| route-27 | FAST | FAST | 0 | PASS |
| route-28 | FAST | FAST | 1 | PASS |
| route-29 | FAST | FAST | 2 | PASS |
| route-30 | QUALITY | QUALITY | 3 | PASS |
| route-31 | QUALITY | QUALITY | 3 | PASS |
| route-32 | FAST | FAST | 2 | PASS |
| route-33 | QUALITY | QUALITY | 4 | PASS |
| route-34 | FAST | FAST | 1 | PASS |
| route-35 | FAST | FAST | 1 | PASS |
| route-36 | QUALITY | QUALITY | 7 | PASS |

## 4. Semantic Cache Evaluation

Category: **OFFLINE**. Attempted cases: 20.

At the configured strict >0.97 threshold on this curated pair set:

| TP | TN | FP | FN | Precision | Recall | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 10 | 0 | 6 | 1.0000 | 0.4000 | 0.7000 |

False positives (review first): None measured.
False negatives: cache-03, cache-04, cache-06, cache-08, cache-09, cache-10.

| Case | Expected reuse | Accepted | Similarity | Distance | Guard compatible | Status |
| --- | --- | --- | --- | --- | --- | --- |
| cache-01 | Yes | Yes | 0.9880 | 0.0120 | Yes | PASS |
| cache-02 | Yes | Yes | 0.9827 | 0.0173 | Yes | PASS |
| cache-03 | Yes | No | 0.7673 | 0.2327 | Yes | FAIL |
| cache-04 | Yes | No | 0.9100 | 0.0900 | Yes | FAIL |
| cache-05 | Yes | Yes | 0.9827 | 0.0173 | Yes | PASS |
| cache-06 | Yes | No | 0.8549 | 0.1451 | Yes | FAIL |
| cache-07 | Yes | Yes | 0.9844 | 0.0156 | Yes | PASS |
| cache-08 | Yes | No | 0.9136 | 0.0864 | Yes | FAIL |
| cache-09 | Yes | No | 0.8530 | 0.1470 | Yes | FAIL |
| cache-10 | Yes | No | 0.9660 | 0.0340 | Yes | FAIL |
| cache-11 | No | No | 0.9322 | 0.0678 | No | PASS |
| cache-12 | No | No | 0.9926 | 0.0074 | No | PASS |
| cache-13 | No | No | 0.9233 | 0.0767 | No | PASS |
| cache-14 | No | No | 0.9437 | 0.0563 | Yes | PASS |
| cache-15 | No | No | 0.9643 | 0.0357 | Yes | PASS |
| cache-16 | No | No | 0.9702 | 0.0298 | No | PASS |
| cache-17 | No | No | 0.8971 | 0.1029 | No | PASS |
| cache-18 | No | No | 0.8747 | 0.1253 | No | PASS |
| cache-19 | No | No | 0.8296 | 0.1704 | No | PASS |
| cache-20 | No | No | 0.4425 | 0.5575 | Yes | PASS |

## 5. RAG Retrieval Evaluation

Category: **OFFLINE**. Attempted cases: 10.

Six synthetic documents (each short enough for one production chunk) and ten queries; success requires both the expected source and unique fact in an accepted retrieved chunk. Ranks are after production distance filtering. This measures retrieval, not answer quality.

| Queries | Hit@1 | Hit@4 | MRR | Misses | Mean retrieval ms |
| --- | --- | --- | --- | --- | --- |
| 10 | 1.0000 | 1.0000 | 1.0000 | 0 | 50.3182 |

| Case | Expected source | Accepted rank | Retrieval ms | Status |
| --- | --- | --- | --- | --- |
| rag-01 | project_meridian.md | 1 | 60.3165 | PASS |
| rag-02 | project_meridian.md | 1 | 35.8754 | PASS |
| rag-03 | project_alpha.md | 1 | 80.1481 | PASS |
| rag-04 | project_alpha.md | 1 | 54.4056 | PASS |
| rag-05 | project_beta.md | 1 | 69.2460 | PASS |
| rag-06 | project_beta.md | 1 | 35.7660 | PASS |
| rag-07 | architecture_notes.md | 1 | 29.7361 | PASS |
| rag-08 | protocol_notes.md | 1 | 33.7922 | PASS |
| rag-09 | storage_notes.md | 1 | 34.8406 | PASS |
| rag-10 | storage_notes.md | 1 | 69.0553 | PASS |

## 6. Controlled FAST vs QUALITY Comparison

Category: **CONTROLLED**. Attempted cases: 4.

| Case | Status | Route | Crew ms | Total ms | Input tokens | Output tokens | Total tokens | DDGS calls | Web results |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| simple-fast | PASS | FAST | 40944.2238 | 42605.1949 | 4090 | 2894 | 6984 | 3 | 3 |
| simple-quality | PASS | QUALITY | 25617.1328 | 27331.8009 | 2768 | 1488 | 4256 | 1 | 1 |
| complex-fast | PASS | FAST | 28696.7443 | 30337.8495 | 3034 | 2937 | 5971 | 1 | 1 |
| complex-quality | PASS | QUALITY | 27290.6603 | 29195.5855 | 3112 | 1531 | 4643 | 1 | 1 |

Deterministic rubric coverage checks concept-group substrings in the final answer. This is a limited content-presence heuristic, not factual correctness or a human-equivalent quality score. Final-answer links can include the backend's appended retrieved-source list.

| Case | Concepts satisfied | Concepts total | Coverage | Source link | Answer characters | Python CPU ms |
| --- | --- | --- | --- | --- | --- | --- |
| simple-fast | 4 | 4 | 1.0000 | Yes | 1769 | 2937.5000 |
| simple-quality | 4 | 4 | 1.0000 | Yes | 2293 | 1968.7500 |
| complex-fast | 6 | 6 | 1.0000 | Yes | 2373 | 1875.0000 |
| complex-quality | 6 | 6 | 1.0000 | Yes | 2288 | 2187.5000 |

| Route | N | Mean total ms | Median total ms |
| --- | --- | --- | --- |
| FAST | 2 | 36471.5222 | 36471.5222 |
| QUALITY | 2 | 28263.6932 | 28263.6932 |

## 7. End-to-End Cache Benchmark

Category: **CONTROLLED**. Attempted cases: 2.

| Case | Status | Route | Crew ms | Total ms | Input tokens | Output tokens | Total tokens | DDGS calls | Web results |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cache-cold | PASS | FAST | 24999.6666 | 32908.5293 | 4370 | 1913 | 6283 | 4 | 4 |
| cache-hit | PASS | FAST | Not reported | 32.9762 | Not reported | Not reported | Not reported | 0 | 0 |

The first lookup in an isolated empty cache is an observable miss labelled `EMPTY` by the backend. Cold means empty cache, not necessarily an unloaded generation model. Both queries use AUTO, FAST models, and web-only scope. Semantic-hit preflight prevents an unintended second Crew generation if reuse fails. RAG is disabled in both requests, so no RAG work was avoided.

| Measurement | Value |
| --- | --- |
| attempted_cases | 2 |
| cold_generation_tokens | 6283 |
| cold_total_ms | 32908.5293 |
| crew_executions_avoided | 1 |
| ddgs_calls_avoided | 4 |
| execution_errors | [] |
| hit_total_ms | 32.9762 |
| latency_reduction | 0.9990 |
| rag_retrievals_avoided | 0 |
| speedup | 997.9479 |
| verified_bypass | Yes |

## 8. RAG Integration Benchmark

Category: **CONTROLLED**. Attempted cases: 1.

| Case | Status | Route | Crew ms | Total ms | Input tokens | Output tokens | Total tokens | DDGS calls | Web results |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rag-integration | PASS | FAST | 38470.9047 | 42266.4961 | 5414 | 2202 | 7616 | 3 | 3 |

| Integration check | Observed |
| --- | --- |
| expected_source_retrieved | Yes |
| fact_in_answer | Yes |
| fact_in_evidence | Yes |
| local_citation | Yes |
| rag_used | Yes |

## 9. Live Web Benchmark

Category: **LIVE-WEB**. Attempted cases: 1.

| Case | Status | Route | Crew ms | Total ms | Input tokens | Output tokens | Total tokens | DDGS calls | Web results |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| live-web-mcp | PASS | FAST | 42850.6473 | 44614.2159 | 2815 | 2466 | 5281 | 1 | 5 |

Real network calls and tool attempts are distinguished; external failure has no successful live-web latency. This single example cannot establish availability or web research quality.

| Case | Real DDGS calls | Blocked attempts | Web search ms |
| --- | --- | --- | --- |
| live-web-mcp | 1 | 0 | 6015.8974 |

## 10. Limitations

- Curated labels are methodology choices; policy conformance does not establish universally correct routing.
- Ten positive and ten negative cache pairs are a small sample. Precision is undefined when no answers are reused; conservative misses reduce recall. Opposing intents not covered by the lexical guard can still be accepted if similarity is high.
- Retrieval uses a small six-document synthetic corpus and top-k four; Hit@K is easier than in a large corpus. Mean retrieval latency excludes ingestion and includes query embedding and local store access.
- One sample per prompt/route and mixed model warmth prevent robust model-speed or quality conclusions. Request latency includes changing local load. Tokens are reported only when the backend measured consistent usage.
- Concept coverage can reward superficial mentions and cannot detect hallucinations, contradicted evidence, or citation entailment. No LLM judge or human evaluation was used.
- One cache-hit sample measures this answer and local state; speedup is not a universal cache performance claim. No API-dollar or total compute saving is estimated.
- A single live-web query is subject to network/service variability. Offline, controlled, and live-web values are not pooled.
- Checkpointed failed/interrupted cases remain terminal. Reservations are conservative if a process stops before invoking research. Ollama generation can make multiple model calls within one Crew; the budget limits research invocations and Crew executions, not individual token requests.

## 11. Reproduction Commands

```powershell
python -m evaluation.cli --suite offline --output validation_logs/phase10
python -m evaluation.cli --suite controlled --output validation_logs/phase10 --resume
python -m evaluation.cli --suite live-web --output validation_logs/phase10 --resume
python -m evaluation.cli --suite all --output validation_logs/phase10 --resume --report-only --report-json evaluation/results/phase10_baseline.json --report-markdown docs/benchmarks/phase10_baseline.md
```

Use a new output directory for a deliberately new run. `--resume` never repeats completed or failed generations. OFFLINE requires the local embedding model. CONTROLLED/LIVE WEB additionally require the two generation models. Model/policy settings remain frozen.
