"""Allowlisted measurements over public execution traces; no LLM judge."""

from hashlib import sha256

from evaluation.metrics import rubric
from evaluation.models import BenchmarkCaseResult
from routing.models import FAST_SYNTHESIS_MODEL, QUALITY_SYNTHESIS_MODEL, SEARCH_MODEL


def evaluate(case_id, suite, result, *, requested_route="auto", concepts=(), urls=()):
    route = result.routing
    models = dict(searcher=route.search_model, analyst=route.analyst_model, writer=route.writer_model) if route else {}
    crew_executions = int(result.timings.crew_ms is not None)
    measured = dict(
        research_status=result.status, cache_status=result.cache.status,
        cache_similarity=result.cache.similarity, cache_distance=result.cache.distance,
        rag_used=result.rag.used, rag_status=result.rag.status, rag_chunks=result.rag.chunk_count,
        ddgs_calls=result.web.total_calls, web_results=result.web.total_results,
        crew_executions=crew_executions, total_ms=result.timings.total_ms,
        total_tokens=result.usage.total_tokens,
        answer_characters=len(result.final_answer) if result.status == "SUCCESS" else None,
        answer_sha256=sha256(result.final_answer.encode()).hexdigest() if result.status == "SUCCESS" else None,
        agent_statuses=[agent.status for agent in result.agents],
        web_statuses=[call.status for call in result.web.calls],
    )
    warnings = []
    # Keep only the backend's fixed warning categories, never exception messages.
    allowed_warnings = {
        "Local RAG unavailable; using web-only research.", "Local RAG retrieval failed; using web-only research.",
        "Semantic cache unavailable; running research.",
        "Expected three public Crew task outputs; some outputs are unavailable.",
    }
    warnings.extend(warning for warning in result.warnings if warning in allowed_warnings)
    if len(warnings) < len(result.warnings):
        warnings.append("Additional backend warnings were reported; their text is omitted.")
    timings = {key: getattr(result.timings, key) for key in (
        "routing_ms", "cache_lookup_ms", "rag_retrieval_ms", "web_search_ms", "crew_ms", "total_ms")}
    usage = {key: getattr(result.usage, key) for key in ("input_tokens", "output_tokens", "total_tokens")}
    valid = result.status == "SUCCESS"
    if requested_route != "auto":
        synthesis = FAST_SYNTHESIS_MODEL if requested_route == "fast" else QUALITY_SYNTHESIS_MODEL
        measured["models_verified"] = bool(route and route.selected_route == requested_route and models == dict(
            searcher=SEARCH_MODEL, analyst=synthesis, writer=synthesis))
        valid &= measured["models_verified"]
    if any(value is not None and value < 0 for value in timings.values()):
        valid = False
        warnings.append("Negative timing detected.")
    if concepts:
        measured["rubric"] = rubric(result.final_answer if valid else "", concepts, urls)
        valid &= measured["rubric"]["nonempty"] and measured["rubric"]["no_execution_error"]
    # A network failure is reported honestly, even if a model writes prose.
    external_failure = suite == "live_web" and bool(result.web.calls) and all(
        call.status == "ERROR" for call in result.web.calls)
    if suite == "live_web" and not result.web.total_results:
        valid = False
    errors = [result.error_type] if result.error_type else []
    if external_failure:
        errors = sorted({call.error_type for call in result.web.calls if call.error_type}) or errors
    if external_failure:
        # Partial failed-work timings are not successful live-web latency.
        timings = {}
        usage = {}
        measured["total_ms"] = None
        measured["total_tokens"] = None
    return BenchmarkCaseResult(case_id, suite,
        "EXTERNAL_UNAVAILABLE" if external_failure else "PASS" if valid else "ERROR" if result.status == "ERROR" else "FAIL",
        duration_ms=result.timings.total_ms if not external_failure else None,
        route=route.selected_route.upper() if route else None, models=models,
        timings=timings, tokens=usage, measured=measured, errors=errors, warnings=warnings)
