# A 3–5 minute demo

Show how evidence, routing, reuse, and measurement fit together. Use real application output and the committed benchmark; no benchmark rerun is required.

## Prepare before presenting

Start Ollama with all three [required models](../README.md#quick-start), then run `python -m streamlit run app.py`. Open the local console and [Phase 10 report](benchmarks/phase10_baseline.md). If doing a live demonstration, allow time for local model loading and web variability. A previously completed real request can stay in the same browser session for presentation; it is not restored automatically after restarting the server.

For a live simple request, disable local knowledge base, enable semantic cache, select Auto, and ask **What is the Model Context Protocol?**. Prepare the local-document segment separately by uploading [project_meridian.md](../evaluation/fixtures/rag/project_meridian.md), explicitly selecting **Add to Knowledge Base**, enabling RAG, and asking **What is Project Meridian's internal protocol codename?**. It is synthetic demo evidence; identify it as such. Expect `Cedar-47` with `[Document: project_meridian.md]` attribution, but verify the actual answer rather than promising generation behavior.

These optional preparation requests are real research and may use DDGS. They are instructions for a later presentation; Phase 11 itself makes no such calls. Do not prepopulate analytics or fabricate trace data for screenshots.

## Presentation script

| Time | Action and explanation |
| --- | --- |
| 0:00–0:30 | Open Research. Explain the three sequential roles and local Ollama inference, with DDGS providing live web evidence. |
| 0:30–1:20 | Show a real simple Auto request or submit it live. Expand Routing Decision: score below 3 selects FAST; Searcher stays on Qwen2.5 3B while synthesis uses Qwen3 1.7B. FAST is a model-capacity label. |
| 1:20–2:00 | Show Execution Summary, Web Search Results, agent task outputs, and Final Research Answer. Explain that task deliverables are observable; hidden reasoning is not shown. Point out source links and available timing/token fields. |
| 2:00–2:50 | Demonstrate explicit Add to Knowledge Base and the synthetic Meridian question, if time allows. Expand RAG Evidence and compare the retrieved codename/source with the final citation. RAG supplies evidence before the Crew runs. |
| 2:50–3:30 | Explain exact/semantic reuse and the one-hour TTL. If showing a live semantic hit, return to the original web-only scope/Auto route within TTL and ask **Explain the Model Context Protocol.** Show the actual status; a hit bypasses Crew/DDGS/retrieval. Otherwise show the recorded cache pair in the report. |
| 3:30–4:10 | Open Analytics: requests, failures/successes, hit rate, routes, latency, and RAG/DDGS usage. Explain query hashes and the absence of research content in SQLite. |
| 4:10–5:00 | Show the benchmark summary: curated router policy-conformance, synthetic retrieval, cache precision/recall, and the single-pair speedup. End with the small-sample surprise: QUALITY was faster than FAST in the two-prompt comparison. |

If a generation is still running, use the architecture/report segment while waiting. Keep the demo within the allotted time by explaining caching from the report rather than issuing extra requests. Semantic reuse is conditional: changed RAG scope/corpus/route, TTL expiry, or insufficient similarity can produce a miss. Do not claim a missed lookup was a hit.

## Screenshot checklist

Capture only real current UI with private content removed or omitted. Use a non-sensitive query and original synthetic fixture. Keep enough context to show which request the evidence belongs to.

1. Research Console overview with Execution Summary, selected models, and Final Research Answer.
2. RAG Evidence expanded with actual source, chunk metadata, and retrieved text.
3. Agent task outputs and final answer, showing the distinction between deliverables and final postprocessing.
4. Analytics dashboard with real operational counts and recent query-hash prefixes.
5. Committed Phase 10 benchmark tables with sample sizes and caveats visible.

Suggested future filenames: `docs/screenshots/research-console.png`, `rag-evidence.png`, `agent-outputs.png`, `analytics.png`, and `phase10-benchmarks.png`. The screenshot directory/files are not prerequisites or existing artifacts. No screenshots were generated for Phase 11; do not substitute fake UI images or test-fixture replay images as evidence of live research.

## Closing explanation

This project demonstrates local inference, evidence integration, scope-aware reuse, transparent routing, and operational measurement. DDGS still requires internet access. The [portfolio](portfolio.md) provides defensible resume bullets and interview answers; the [architecture](architecture.md) provides implementation detail.
