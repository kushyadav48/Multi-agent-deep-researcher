import pytest

from evaluation.metrics import cache_metrics, cache_speedup, latency_summary, rag_metrics, routing_metrics, rubric


def test_router_confusion_matrix_and_accuracy():
    rows = [dict(expected_route=e, actual_route=a) for e, a in (
        ("FAST", "FAST"), ("FAST", "QUALITY"), ("QUALITY", "FAST"), ("QUALITY", "QUALITY"))]
    result = routing_metrics(rows)
    assert result["accuracy"] == .5 and result["correct"] == result["incorrect"] == 2
    assert result["confusion_matrix"] == {"FAST": {"FAST": 1, "QUALITY": 1}, "QUALITY": {"FAST": 1, "QUALITY": 1}}
    assert routing_metrics([])["accuracy"] is None


def test_cache_classification_ratios_and_failure_ids():
    rows = [dict(case_id=str(i), expected_reuse=e, accepted=a) for i, (e, a) in enumerate(
        ((True, True), (True, False), (False, True), (False, False)))]
    result = cache_metrics(rows)
    assert all(result[key] == 1 for key in ("TP", "TN", "FP", "FN"))
    assert result["precision"] == result["recall"] == result["accuracy"] == .5
    assert result["false_positives"] == ["2"] and result["false_negatives"] == ["1"]
    assert all(cache_metrics([])[key] is None for key in ("precision", "recall", "accuracy"))


def test_rag_hit_rank_and_mean_latency():
    result = rag_metrics([dict(rank=rank, retrieval_ms=ms) for rank, ms in ((1, 10), (2, 20), (None, 30))], 4)
    assert result["hit_at_1"] == pytest.approx(1 / 3)
    assert result["hit_at_k"] == pytest.approx(2 / 3) and result["mrr"] == .5
    assert result["misses"] == 1 and result["mean_retrieval_ms"] == 20
    assert rag_metrics([], 4)["mrr"] is None


def test_concept_groups_count_alternatives_once_and_missing_usage():
    result = rubric("DENSE retrieval is semantic retrieval. A CLIENT uses tools. https://example.org", [
        ["dense retrieval", "semantic retrieval"], ["client"], ["server"]], ["https://example.org"])
    assert result["concepts_satisfied"] == 2 and result["coverage"] == pytest.approx(2 / 3)
    assert result["expected_source_link"] and result["nonempty"] and result["no_execution_error"]
    assert not rubric("Error: unavailable", [["client"]])["no_execution_error"]
    assert not rubric("", [["client"]])["nonempty"]
    assert rubric("answer", [])['coverage'] is None


def test_speedup_requires_actual_semantic_bypass():
    cold = dict(cache_status="EMPTY", crew_executions=1, total_ms=1000, ddgs_calls=2, total_tokens=30)
    hit = dict(cache_status="SEMANTIC_HIT", crew_executions=0, total_ms=10, ddgs_calls=0, rag_used=False)
    result = cache_speedup(cold, hit)
    assert result["speedup"] == 100 and result["latency_reduction"] == .99
    assert result["crew_executions_avoided"] == 1 and result["ddgs_calls_avoided"] == 2
    assert result["rag_retrievals_avoided"] == 0 and result["cold_generation_tokens"] == 30
    assert cache_speedup(cold, dict(hit, crew_executions=1))["speedup"] is None
    assert cache_speedup(cold, dict(hit, total_ms=0))["speedup"] is None
    assert cache_speedup(cold, dict(hit, cache_status="MISS"))["speedup"] is None
    assert latency_summary([1, 3]) == dict(n=2, mean_ms=2, median_ms=2)
