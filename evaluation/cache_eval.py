"""Real cache classification, isolated per query pair."""

from time import perf_counter, time

from evaluation.models import BenchmarkCaseResult
from semantic_cache.models import CacheScope
from semantic_cache.service import SemanticCacheService, queries_compatible
from semantic_cache.store import ChromaCacheStore


def evaluate(case, provider, path):
    started = perf_counter()
    cache = SemanticCacheService(provider, ChromaCacheStore(path))
    scope = CacheScope(use_rag=False)
    cache.save(case["first"], "Nonsensitive benchmark cached answer.", scope)
    lookup_started = perf_counter()
    lookup = cache.lookup(case["second"], scope)
    lookup_ms = (perf_counter() - lookup_started) * 1000
    # Misses don't expose candidate scores. Read the same production candidate,
    # with the already computed vector, without another embedding request.
    candidate = (cache.store.search(lookup.embedding, scope, time(), cache.ttl_seconds)
                 if lookup.embedding is not None else None)
    similarity = lookup.hit.similarity if lookup.hit else candidate[1] if candidate else None
    accepted = lookup.hit is not None
    return BenchmarkCaseResult(case["id"], "cache_pairs",
        "PASS" if accepted == case["expected_reuse"] else "FAIL",
        duration_ms=(perf_counter() - started) * 1000,
        measured=dict(expected_reuse=case["expected_reuse"], accepted=accepted,
                      similarity=similarity, distance=1 - similarity if similarity is not None else None,
                      hit_kind=lookup.hit.kind if lookup.hit else None, lookup_ms=lookup_ms,
                      compatible=queries_compatible(case["first"], case["second"])))
