"""Call the real deterministic router, without generation or embeddings."""

from time import perf_counter

from evaluation.models import BenchmarkCaseResult
from routing.router import route_query


def evaluate(case):
    started = perf_counter()
    decision = route_query(case["query"])
    actual = decision.selected_route.value.upper()
    return BenchmarkCaseResult(case["id"], "router", "PASS" if actual == case["expected_route"] else "FAIL",
        duration_ms=(perf_counter() - started) * 1000, route=actual,
        measured=dict(expected_route=case["expected_route"], actual_route=actual, score=decision.score))
