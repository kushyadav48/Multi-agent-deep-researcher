"""Generate JSON and Markdown strictly from checkpointed measurements."""

import json
from pathlib import Path


def value(item):
    if item is None:
        return "Not reported"
    if isinstance(item, bool):
        return "Yes" if item else "No"
    if isinstance(item, float):
        return f"{item:.4f}"
    return str(item).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                      ["| " + " | ".join(value(item) for item in row) + " |" for row in rows])


def markdown(run):
    suites = {suite.name: suite for suite in run.suites}
    lines = [f"# Phase 10 baseline ({run.benchmark_version})", "", "## 1. Benchmark Environment", "",
        f"- Timestamp (UTC): {run.timestamp}", f"- Production checkpoint commit: `{run.git_commit}`",
        f"- Harness/dataset SHA-256: `{run.source_hash}`",
        "- Harness files were evaluated before their Phase 10 commit; the hash identifies that exact methodology.",
        f"- Python: {run.environment['python']}; platform: {run.environment['platform']}",
        f"- Research invocations started: {run.research_invocations}; research reservations: {run.research_reservations}; observed Crew executions: {run.crew_executions}; Crew reservations: {run.crew_reservations}.",
        "", table(["Configuration", "Value"], sorted(run.configuration.items())), "",
        "## 2. Methodology", "",
        "OFFLINE uses the real deterministic router, production cache lookup, and production RAG retrieval. Cache and RAG use real local embeddings, but no generation or DDGS network.",
        "CONTROLLED EVIDENCE uses the canonical detailed research API and patches only `ddgs.ddgs.DDGS.text`. Both routes receive identical fixed evidence for each of two prompts. Cache reuse and local evidence propagation are separate cases.",
        "LIVE WEB has one research case and permits at most one real DDGS boundary call. Any additional tool attempt is blocked and reported. No failed case is retried automatically.",
        "All Chroma/cache/metrics stores are isolated under the ignored benchmark output directory. Atomic checkpoints retain completed cases. Interrupted reservations are never automatically retried. No generated answers, agent outputs, prompts, vectors, credentials, or verbose logs appear in this report or result JSON.",
        "Durations are milliseconds from the production trace (or `perf_counter` for offline cases). Research trace timings exclude harness setup, semantic-hit preflight, event draining, and metrics persistence. Python process CPU time excludes the Ollama service and GPU. It is coordinator work, not total inference resource usage. Small synthesis groups report N, mean, and median; no p95 is claimed. Each prompt/route has one sample, in FAST then QUALITY order, without generation warm-up or a fixed random seed.",
        "The frozen MCP evidence paraphrases the [official architecture documentation](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture). Hybrid evidence paraphrases [Microsoft's hybrid-search documentation](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview); its deployment trade-off recommendation is explicitly a benchmark engineering inference. Synthetic project facts are original benchmark content.", ""]
    if run.environment.get("local_models"):
        lines.extend([table(["Local model", "Digest", "Bytes", "Quantization"], [[
            model["name"], model.get("digest"), model.get("size_bytes"), model.get("quantization")]
            for model in run.environment["local_models"]]), ""])
    sections = [
        ("router", "3. Router Policy Evaluation"), ("cache_pairs", "4. Semantic Cache Evaluation"),
        ("rag_retrieval", "5. RAG Retrieval Evaluation"), ("synthesis", "6. Controlled FAST vs QUALITY Comparison"),
        ("cache_e2e", "7. End-to-End Cache Benchmark"), ("rag_integration", "8. RAG Integration Benchmark"),
        ("live_web", "9. Live Web Benchmark"),
    ]
    for name, title in sections:
        lines.extend([f"## {title}", ""])
        suite = suites.get(name)
        if not suite:
            lines.extend(["Not run.", ""])
            continue
        lines.extend([f"Category: **{suite.category}**. Attempted cases: {len(suite.cases)}.", ""])
        metrics = suite.metrics
        if name == "router":
            lines.extend([f"Curated policy-conformance accuracy: {value(metrics['accuracy'])} ({metrics['correct']}/{metrics['total']}). This set checks v1 behavior, including bare comparisons that remain FAST and lexical false alarms such as 'recommendation engine'; it is not general routing intelligence.", "",
                table(["Expected / Actual", "FAST", "QUALITY"], [[expected, *metrics['confusion_matrix'][expected].values()] for expected in ("FAST", "QUALITY")]), "",
                table(["Case", "Expected", "Actual", "Score", "Status"], [[c.case_id, c.measured.get('expected_route'), c.route, c.measured.get('score'), c.status] for c in suite.cases])])
        elif name == "cache_pairs":
            lines.extend([f"At the configured strict >{run.configuration['cache_similarity_threshold']} threshold on this curated pair set:", "",
                table(["TP", "TN", "FP", "FN", "Precision", "Recall", "Accuracy"], [[metrics[key] for key in ("TP", "TN", "FP", "FN", "precision", "recall", "accuracy")]]), "",
                f"False positives (review first): {', '.join(metrics['false_positives']) or 'None measured'}.",
                f"False negatives: {', '.join(metrics['false_negatives']) or 'None measured'}.", "",
                table(["Case", "Expected reuse", "Accepted", "Similarity", "Distance", "Guard compatible", "Status"], [
                    [c.case_id, *[c.measured.get(key) for key in ('expected_reuse', 'accepted', 'similarity', 'distance', 'compatible')], c.status] for c in suite.cases])])
        elif name == "rag_retrieval":
            lines.extend(["Six synthetic documents (each short enough for one production chunk) and ten queries; success requires both the expected source and unique fact in an accepted retrieved chunk. Ranks are after production distance filtering. This measures retrieval, not answer quality.", "",
                table(["Queries", "Hit@1", f"Hit@{metrics['top_k']}", "MRR", "Misses", "Mean retrieval ms"], [[metrics[key] for key in ('total', 'hit_at_1', 'hit_at_k', 'mrr', 'misses', 'mean_retrieval_ms')]]), "",
                table(["Case", "Expected source", "Accepted rank", "Retrieval ms", "Status"], [[c.case_id, *[c.measured.get(key) for key in ('expected_source', 'rank', 'retrieval_ms')], c.status] for c in suite.cases])])
        else:
            lines.extend([table(["Case", "Status", "Route", "Crew ms", "Total ms", "Input tokens", "Output tokens", "Total tokens", "DDGS calls", "Web results"], [[
                c.case_id, c.status, c.route, c.timings.get('crew_ms'), c.timings.get('total_ms'),
                c.tokens.get('input_tokens'), c.tokens.get('output_tokens'), c.tokens.get('total_tokens'),
                c.measured.get('ddgs_calls'), c.measured.get('web_results')] for c in suite.cases]), ""])
            if name == "synthesis":
                lines.extend(["Deterministic rubric coverage checks concept-group substrings in the final answer. This is a limited content-presence heuristic, not factual correctness or a human-equivalent quality score. Final-answer links can include the backend's appended retrieved-source list.", "",
                    table(["Case", "Concepts satisfied", "Concepts total", "Coverage", "Source link", "Answer characters", "Python CPU ms"], [[c.case_id,
                        *[c.measured.get('rubric', {}).get(key) for key in ('concepts_satisfied', 'concepts_total', 'coverage', 'expected_source_link')],
                        c.measured.get('answer_characters'), c.measured.get('python_process_cpu_ms')] for c in suite.cases]), "",
                    table(["Route", "N", "Mean total ms", "Median total ms"], [[route, *[metrics[route][key] for key in ('n', 'mean_ms', 'median_ms')]] for route in ('FAST', 'QUALITY')])])
            if name == "cache_e2e":
                lines.extend(["The first lookup in an isolated empty cache is an observable miss labelled `EMPTY` by the backend. Cold means empty cache, not necessarily an unloaded generation model. Both queries use AUTO, FAST models, and web-only scope. Semantic-hit preflight prevents an unintended second Crew generation if reuse fails. RAG is disabled in both requests, so no RAG work was avoided.", "",
                              table(["Measurement", "Value"], sorted(metrics.items()))])
            if name == "rag_integration":
                for case in suite.cases:
                    lines.append(table(["Integration check", "Observed"], sorted(case.measured.get('integration_checks', {}).items())))
            if name == "live_web":
                lines.extend(["Real network calls and tool attempts are distinguished; external failure has no successful live-web latency. This single example cannot establish availability or web research quality.", "",
                    table(["Case", "Real DDGS calls", "Blocked attempts", "Web search ms"], [[c.case_id, c.measured.get('real_ddgs_calls'), c.measured.get('blocked_ddgs_calls'), c.timings.get('web_search_ms')] for c in suite.cases])])
        for case in suite.cases:
            if case.errors:
                lines.append(f"- {case.case_id}: {case.status}; error types: {', '.join(case.errors)}.")
            for warning in case.warnings:
                lines.append(f"- {case.case_id}: {warning}")
        lines.append("")
    lines.extend(["## 10. Limitations", "",
        "- Curated labels are methodology choices; policy conformance does not establish universally correct routing.",
        "- Ten positive and ten negative cache pairs are a small sample. Precision is undefined when no answers are reused; conservative misses reduce recall. Opposing intents not covered by the lexical guard can still be accepted if similarity is high.",
        "- Retrieval uses a small six-document synthetic corpus and top-k four; Hit@K is easier than in a large corpus. Mean retrieval latency excludes ingestion and includes query embedding and local store access.",
        "- One sample per prompt/route and mixed model warmth prevent robust model-speed or quality conclusions. Request latency includes changing local load. Tokens are reported only when the backend measured consistent usage.",
        "- Concept coverage can reward superficial mentions and cannot detect hallucinations, contradicted evidence, or citation entailment. No LLM judge or human evaluation was used.",
        "- One cache-hit sample measures this answer and local state; speedup is not a universal cache performance claim. No API-dollar or total compute saving is estimated.",
        "- A single live-web query is subject to network/service variability. Offline, controlled, and live-web values are not pooled.",
        "- Checkpointed failed/interrupted cases remain terminal. Reservations are conservative if a process stops before invoking research. Ollama generation can make multiple model calls within one Crew; the budget limits research invocations and Crew executions, not individual token requests.", "",
        "## 11. Reproduction Commands", "", "```powershell",
        "python -m evaluation.cli --suite offline --output validation_logs/phase10",
        "python -m evaluation.cli --suite controlled --output validation_logs/phase10 --resume",
        "python -m evaluation.cli --suite live-web --output validation_logs/phase10 --resume",
        "python -m evaluation.cli --suite all --output validation_logs/phase10 --resume --report-only --report-json evaluation/results/phase10_baseline.json --report-markdown docs/benchmarks/phase10_baseline.md",
        "```", "", "Use a new output directory for a deliberately new run. `--resume` never repeats completed or failed generations. OFFLINE requires the local embedding model. CONTROLLED/LIVE WEB additionally require the two generation models. Model/policy settings remain frozen.", ""])
    return "\n".join(lines)


def write_reports(run, json_path, markdown_path):
    for path, content in ((Path(json_path), json.dumps(run.to_dict(), indent=2, allow_nan=False) + "\n"),
                          (Path(markdown_path), markdown(run))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
