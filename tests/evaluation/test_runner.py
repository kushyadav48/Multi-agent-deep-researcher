from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from evaluation import cache_eval, rag_eval, router_eval, synthesis_eval
from evaluation.datasets import load
from evaluation.models import BenchmarkCaseResult, BenchmarkRun
from evaluation.reporting import markdown, write_reports
from evaluation.runner import BudgetExceeded, CACHE_QUERY, LIVE_QUERY, Runner, output_lock
from observability.models import CacheTrace, RAGTrace, RoutingTrace, Timings, Usage, WebSearchCall, WebSearchResult, WebTrace
from rag.context import retrieve_document_context
from routing.router import route_query
from tests.ui.fixtures import execution_result


class FakeEmbeddings:
    model = "qwen3-embedding:0.6b"

    def embed_text(self, text):
        text = text.casefold()
        groups = (("model context protocol", "mcp"), ("meridian",), ("alpha",), ("beta",),
                  ("replica", "architecture"), ("integrity", "quartz"), ("archive", "storage", "retention"))
        return [float(any(term in text for term in group)) for group in groups] + [.01]

    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]


def fake_research(query, **kwargs):
    import agents
    from ddgs.ddgs import DDGS
    from observability.models import RAGEvidence
    decision = route_query(query, kwargs["model_route"])
    result = execution_result()
    result.query = query
    result.routing = RoutingTrace(decision.requested_mode.value, decision.selected_route.value,
        "SIMPLE" if decision.selected_route.value == "fast" else "COMPLEX", decision.score, 3,
        decision.reasons, decision.policy_version, decision.search_model, decision.synthesis_model, decision.synthesis_model)
    result.timings.total_ms = 1000 if decision.selected_route.value == "fast" else 2000
    cache = kwargs.get("cache_service")
    if cache:
        scope = agents.research_cache_scope(None, use_rag=False, top_k=4, max_distance=.6, routing_decision=decision)
        lookup = cache.lookup(query, scope)
        if lookup.hit:
            result.cache = CacheTrace("SEMANTIC_HIT", lookup.hit.similarity)
            result.final_answer = lookup.hit.entry.answer
            result.timings = Timings(1, 2, total_ms=10)
            result.web = WebTrace()
            result.usage = Usage()
            return result
        result.cache = CacheTrace("EMPTY")
    rows = DDGS().text(query, max_results=5)
    result.web = WebTrace([WebSearchCall(query, [WebSearchResult(row["title"], row["href"], row["body"]) for row in rows], 1, "SUCCESS")])
    result.final_answer = "Model Context Protocol: a client uses a server with tools. " + rows[0]["href"]
    if kwargs["use_rag"]:
        evidence = []
        retrieve_document_context(query, kwargs["rag_service"], observer=lambda status, chunks: evidence.extend(chunks))
        result.rag = RAGTrace(True, "USED", [RAGEvidence(chunk.source, chunk.page, chunk.chunk_index,
                                                     chunk.distance, chunk.text) for chunk in evidence])
        result.final_answer = "Cedar-47 [Document: project_meridian.md]"
    if cache:
        cache.save(query, result.final_answer, scope)
    return result


def runner(path, **kwargs):
    return Runner(path, research=fake_research, provider_factory=FakeEmbeddings,
                  progress=lambda *args, **kw: None, **kwargs)


def test_real_router_offline_without_research():
    for case in load("routing"):
        result = router_eval.evaluate(case)
        assert result.measured["actual_route"] in ("FAST", "QUALITY")


def test_real_cache_and_rag_apis_with_fake_embeddings(tmp_path):
    result = cache_eval.evaluate(load("semantic_cache")[0], FakeEmbeddings(), tmp_path / "cache")
    assert result.status == "PASS" and result.measured["similarity"] > .97
    service = rag_eval.prepare(FakeEmbeddings(), tmp_path / "rag")
    result = rag_eval.evaluate(load("rag")[0], service)
    assert result.measured["rank"] == 1 and service.count() == 6


def test_controlled_suite_is_seven_requests_six_crews_and_isolated(tmp_path):
    calls = Mock(side_effect=fake_research)
    run = Runner(tmp_path / "out", research=calls, provider_factory=FakeEmbeddings, progress=lambda *a, **kw: None)
    result = run.run("controlled")
    assert calls.call_count == result.research_invocations == 7
    assert result.crew_executions == result.crew_reservations == 6
    assert {suite.name: len(suite.cases) for suite in result.suites} == dict(synthesis=4, cache_e2e=2, rag_integration=1)
    for prompt in ("simple", "complex"):
        fast, quality = run.find(f"{prompt}-fast"), run.find(f"{prompt}-quality")
        assert fast.measured["fixed_evidence_verified"] and quality.measured["fixed_evidence_verified"]
        assert fast.measured["fixed_evidence_sha256"] == quality.measured["fixed_evidence_sha256"]
    assert run.find("cache-hit").measured["answer_matches_cold"]
    assert run.find("cache-hit").status == "PASS"
    assert "fixed_evidence_verified" not in run.find("cache-hit").measured
    assert run.suite("cache_e2e").metrics["speedup"] == 100
    assert all(run.find("rag-integration").measured["integration_checks"].values())
    for call in calls.call_args_list:
        assert tmp_path / "out" / "state" in call.kwargs["metrics_store"].path.parents
        for name in ("rag_service", "cache_service"):
            if name in call.kwargs:
                store = getattr(call.kwargs[name], "vector_store", getattr(call.kwargs[name], "store", None))
                assert tmp_path / "out" / "state" in store.path.parents


def test_resume_skips_success_failures_and_interruptions(tmp_path):
    first = runner(tmp_path / "out")
    first.case("saved", "synthesis", lambda: BenchmarkCaseResult("saved", "synthesis", "PASS"))
    first.case("failed", "synthesis", lambda: (_ for _ in ()).throw(RuntimeError("SECRET")), expensive=True)
    # Simulate a crash after reserving a call, before committing its outcome.
    first.suite("synthesis").cases.append(BenchmarkCaseResult("interrupted", "synthesis", "IN_PROGRESS"))
    first.run_result.research_reservations += 1
    first.save()
    resumed = runner(tmp_path / "out", resume=True)
    action = Mock()
    for identity in ("saved", "failed", "interrupted"):
        resumed.case(identity, "synthesis", action, expensive=True)
    action.assert_not_called()
    assert resumed.find("interrupted").status == "INTERRUPTED"
    assert resumed.find("failed").errors == ["RuntimeError"]
    assert "SECRET" not in resumed.checkpoint.read_text()


@pytest.mark.parametrize("field,limit", [("research_reservations", 8), ("crew_reservations", 7)])
def test_expensive_caps_fail_before_action(tmp_path, field, limit):
    run = runner(tmp_path / "out")
    setattr(run.run_result, field, limit)
    action = Mock()
    with pytest.raises(BudgetExceeded):
        run.case("extra", "synthesis", action, expensive=True)
    action.assert_not_called()


def test_live_web_has_one_case_and_only_one_network_boundary_call(tmp_path):
    def fake(query, **kwargs):
        from ddgs.ddgs import DDGS
        DDGS().text(query)
        with pytest.raises(Exception) as error:
            DDGS().text("second forbidden query")
        assert type(error.value).__name__ == "LiveWebBudgetExceeded"
        return execution_result()
    with patch("ddgs.ddgs.DDGS.text", return_value=[]) as network:
        calls = Mock(side_effect=fake)
        run = Runner(tmp_path / "out", research=calls, provider_factory=FakeEmbeddings, progress=lambda *a, **kw: None)
        run.run("live-web")
        network.assert_called_once()
    assert calls.call_count == 1 and calls.call_args.args == (LIVE_QUERY,)
    case = run.find("live-web-mcp")
    assert case.measured["real_ddgs_calls"] == case.measured["blocked_ddgs_calls"] == 1


def test_semantic_miss_preflight_never_generates_a_second_answer(tmp_path):
    run = runner(tmp_path / "out")
    run.run("controlled")
    with patch.object(run.cache(), "lookup", return_value=Mock(hit=None)), patch("agents.run_research_detailed") as research:
        result = run.execute("cache-hit", "cache_e2e", "different query", use_cache=True, preflight_hit=True)
    research.assert_not_called()
    assert result.status == "FAIL" and result.measured["research_not_invoked"]


def test_checkpoint_mismatch_overwrite_and_production_state_are_rejected(tmp_path):
    runner(tmp_path / "out")
    with pytest.raises(ValueError):
        runner(tmp_path / "out")
    with patch("evaluation.runner.source_hash", return_value="different"):
        with pytest.raises(ValueError):
            runner(tmp_path / "out", resume=True)
    from evaluation.runner import PROJECT
    with pytest.raises(ValueError):
        runner(PROJECT / "data" / "metrics" / "bench")


def test_source_hash_serialization_report_and_privacy(tmp_path):
    run = runner(tmp_path / "out")
    result = execution_result()
    result.final_answer = "PRIVATE ANSWER AND SYSTEM PROMPT"
    result.warnings = ["PRIVATE CREDENTIAL"]
    result.messages = "PRIVATE HIDDEN REASONING"
    result.embeddings = [123456.789]
    run.case("privacy", "synthesis", lambda: synthesis_eval.evaluate("privacy", "synthesis", result))
    serialized = json.dumps(run.run_result.to_dict())
    assert "PRIVATE" not in serialized and "123456.789" not in serialized
    restored = BenchmarkRun.from_dict(json.loads(serialized))
    assert restored.to_dict() == run.run_result.to_dict()
    write_reports(restored, tmp_path / "results.json", tmp_path / "report.md")
    assert "PRIVATE" not in (tmp_path / "report.md").read_text()
    assert all(title in markdown(restored) for title in ("Benchmark Environment", "Live Web Benchmark", "Limitations", "Reproduction Commands"))


def test_network_error_has_external_status_without_success_latency():
    result = execution_result("error")
    result.web = WebTrace([WebSearchCall("query", [], 10, "ERROR", "TimeoutError")])
    case = synthesis_eval.evaluate("web", "live_web", result)
    assert case.status == "EXTERNAL_UNAVAILABLE" and case.errors == ["TimeoutError"]
    assert case.duration_ms is None and not case.timings and not case.tokens


def test_output_lock_releases_for_later_resume(tmp_path):
    with output_lock(tmp_path / "out"):
        with pytest.raises(OSError):
            with output_lock(tmp_path / "out"):
                pass
    with output_lock(tmp_path / "out"):
        pass


def test_offline_selection_never_calls_research_or_generation(tmp_path):
    research = Mock()
    run = Runner(tmp_path / "out", research=research, provider_factory=FakeEmbeddings, progress=lambda *a, **kw: None)
    def cache_case(item, *args):
        return BenchmarkCaseResult(item["id"], "cache_pairs", "PASS",
            measured=dict(expected_reuse=item["expected_reuse"], accepted=item["expected_reuse"]))
    def rag_case(item, *args):
        return BenchmarkCaseResult(item["id"], "rag_retrieval", "PASS", measured=dict(rank=1, retrieval_ms=1))
    with patch("evaluation.cache_eval.evaluate", side_effect=cache_case) as cache, \
         patch("evaluation.rag_eval.prepare", return_value=Mock()) as prepare, \
         patch("evaluation.rag_eval.evaluate", side_effect=rag_case) as rag:
        result = run.run("offline")
    research.assert_not_called()
    assert result.research_invocations == result.crew_executions == 0
    assert {suite.name: len(suite.cases) for suite in result.suites} == dict(router=36, cache_pairs=20, rag_retrieval=10)
    assert cache.call_count == 20 and rag.call_count == 10 and prepare.call_count == 1
