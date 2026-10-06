"""Public Phase 8 traces for UI replay; no backend or model execution."""

from observability.models import (
    AgentOutput, CacheTrace, RAGEvidence, RAGTrace, ResearchExecutionResult,
    RoutingTrace, Timings, Usage, WebSearchCall, WebSearchResult, WebTrace,
)


def execution_result(kind="simple"):
    complex_request = kind == "complex"
    synthesis = "ollama/qwen2.5:3b" if complex_request else "ollama/qwen3:1.7b"
    result = ResearchExecutionResult(
        request_id=f"fixture-{kind}", query="PRIVATE QUERY: compare protocols" if complex_request else "What is MCP?",
        started_at="2026-10-07T08:00:00+00:00", finished_at="2026-10-07T08:00:02+00:00",
        requested_route_mode="auto", metrics_persisted=True,
        routing=RoutingTrace("auto", "quality" if complex_request else "fast",
                             "COMPLEX" if complex_request else "SIMPLE", 7 if complex_request else 0,
                             3, ("Comparison requested (+2)", "Recommendation requested (+3)")
                             if complex_request else ("Auto: score 0 < 3; selected fast",),
                             "v1", "ollama/qwen2.5:3b", synthesis, synthesis),
        cache=CacheTrace("MISS", threshold=.97),
        web=WebTrace([WebSearchCall("MCP public protocol", [
            WebSearchResult("MCP documentation", "https://modelcontextprotocol.io/", "Open protocol snippet")
        ], 125, "SUCCESS")]),
        agents=[AgentOutput(name, task, model, "COMPLETED", output, None)
                for name, task, model, output in [
                    ("Web Searcher", "web_search", "ollama/qwen2.5:3b", "Observable search deliverable"),
                    ("Research Analyst", "analysis", synthesis, "Observable analyst deliverable"),
                    ("Technical Writer", "writing", synthesis, "Raw writer deliverable"),
                ]],
        timings=Timings(1, 2, None, 125, 1500, 1600),
        usage=Usage(20, 10, 30), final_answer="## Final answer\nPostprocessed final research answer",
    )
    if complex_request:
        result.rag = RAGTrace(True, "USED", [
            RAGEvidence("architecture.pdf", 6, 3, .18, "Paged PDF evidence\n" + "Useful context. " * 150),
            RAGEvidence("notes.md", None, 0, .12, "Nonpaged Markdown evidence"),
        ], 35)
        result.timings.rag_retrieval_ms = 35
    if kind in ("semantic_hit", "exact_hit"):
        result.cache = CacheTrace("SEMANTIC_HIT" if kind == "semantic_hit" else "EXACT_HIT",
                                  .988 if kind == "semantic_hit" else 1, .012 if kind == "semantic_hit" else 0,
                                  .97)
        result.rag = RAGTrace(True, "BYPASSED", reason="semantic_cache_hit")
        result.web = WebTrace()
        for agent in result.agents:
            agent.status, agent.output_text, agent.reason = "NOT_EXECUTED", None, "semantic_cache_hit"
        result.timings = Timings(1, 2, total_ms=3)
        result.usage = Usage()
    if kind == "error":
        result.status = "ERROR"
        result.error_type = "RuntimeError"
        result.errors = ["RuntimeError"]
        result.final_answer = "Error: secret credential traceback must not be displayed"
        result.warnings = ["Local RAG unavailable; using web-only research."]
        result.agents[1].status = "ERROR"
        result.agents[1].output_text = None
        result.agents[1].reason = "crew_failed"
    return result
