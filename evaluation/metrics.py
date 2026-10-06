"""Limited deterministic metrics, with undefined ratios represented by None."""

from statistics import mean, median


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def routing_metrics(rows):
    matrix = {expected: {actual: 0 for actual in ("FAST", "QUALITY")} for expected in ("FAST", "QUALITY")}
    for row in rows:
        matrix[row["expected_route"]][row["actual_route"]] += 1
    correct = sum(matrix[value][value] for value in matrix)
    return dict(total=len(rows), correct=correct, incorrect=len(rows) - correct,
                accuracy=ratio(correct, len(rows)), confusion_matrix=matrix)


def cache_metrics(rows):
    counts = dict(TP=0, TN=0, FP=0, FN=0)
    for row in rows:
        key = ("TP" if row["accepted"] else "FN") if row["expected_reuse"] else (
            "FP" if row["accepted"] else "TN")
        counts[key] += 1
    return dict(**counts, total=len(rows), precision=ratio(counts["TP"], counts["TP"] + counts["FP"]),
                recall=ratio(counts["TP"], counts["TP"] + counts["FN"]),
                accuracy=ratio(counts["TP"] + counts["TN"], len(rows)),
                false_positives=[r["case_id"] for r in rows if r["accepted"] and not r["expected_reuse"]],
                false_negatives=[r["case_id"] for r in rows if not r["accepted"] and r["expected_reuse"]])


def rag_metrics(rows, top_k):
    ranks = [row["rank"] for row in rows]
    latencies = [row["retrieval_ms"] for row in rows]
    hits = sum(rank is not None and rank <= top_k for rank in ranks)
    return dict(total=len(rows), top_k=top_k, hit_at_1=ratio(sum(rank == 1 for rank in ranks), len(rows)),
                hit_at_k=ratio(hits, len(rows)), mrr=ratio(sum(1 / rank for rank in ranks if rank), len(rows)),
                source_hits=hits, misses=len(rows) - hits,
                mean_retrieval_ms=mean(latencies) if latencies else None)


def rubric(answer, concept_groups, urls=()):
    normalized = " ".join(answer.casefold().split())
    satisfied = [any(" ".join(term.casefold().split()) in normalized for term in group) for group in concept_groups]
    nonempty = bool(answer.strip())
    no_error = not normalized.startswith(("error:", "error occurred", "traceback", "failed:"))
    return dict(concepts_satisfied=sum(satisfied), concepts_total=len(satisfied),
                coverage=ratio(sum(satisfied), len(satisfied)), concept_checks=satisfied,
                nonempty=nonempty, no_execution_error=no_error,
                expected_source_link=any(url in answer for url in urls) if urls else None)


def cache_speedup(cold, hit):
    """Only label speedup when actual semantic bypass is observable."""
    valid = (cold["cache_status"] in ("EMPTY", "MISS") and cold["crew_executions"] == 1
             and hit["cache_status"] == "SEMANTIC_HIT" and hit["crew_executions"] == 0
             and hit["ddgs_calls"] == 0 and not hit["rag_used"]
             and cold["total_ms"] is not None and hit["total_ms"] is not None and hit["total_ms"] > 0)
    return dict(verified_bypass=valid, cold_total_ms=cold["total_ms"], hit_total_ms=hit["total_ms"],
                speedup=cold["total_ms"] / hit["total_ms"] if valid else None,
                latency_reduction=1 - hit["total_ms"] / cold["total_ms"] if valid and cold["total_ms"] > 0 else None,
                crew_executions_avoided=cold["crew_executions"] if valid else None,
                ddgs_calls_avoided=cold["ddgs_calls"] if valid else None,
                rag_retrievals_avoided=0 if valid else None,
                cold_generation_tokens=cold.get("total_tokens") if valid else None)


def latency_summary(values):
    return dict(n=len(values), mean_ms=mean(values) if values else None,
                median_ms=median(values) if values else None)
