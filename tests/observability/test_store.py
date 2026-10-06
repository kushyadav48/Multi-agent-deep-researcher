from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import sqlite3

import pytest

from observability.models import WebSearchCall, WebSearchResult
from observability.recorder import ExecutionRecorder
from observability.store import MetricsStore, SCHEMA_VERSION
from routing import route_query


def record(store, query, latency, *, status='SUCCESS', cache='MISS', mode='fast', rag=False):
    recorder = ExecutionRecorder(query, use_rag=rag, use_cache=True, model_route=mode)
    recorder.routing(route_query(query, mode))
    result = recorder.result
    result.finished_at = result.started_at
    result.status = status
    result.error_type = 'RuntimeError' if status == 'ERROR' else None
    result.cache.status = cache
    result.cache.similarity = .99 if cache.endswith('HIT') else None
    result.timings.total_ms = latency
    result.timings.routing_ms = .1
    result.timings.crew_ms = 20 if cache == 'MISS' else None
    result.rag.status = 'USED' if rag else 'DISABLED'
    if cache == 'MISS':
        result.web.calls = [WebSearchCall('PRIVATE SEARCH', [WebSearchResult('title', 'url', 'PRIVATE SNIPPET')],
                                        2, 'SUCCESS')]
    result.final_answer = 'PRIVATE FINAL ANSWER'
    store.record(result)
    return result


def test_schema_allowlist_privacy_and_recreation(tmp_path):
    path = tmp_path / 'nested' / 'metrics.db'
    store = MetricsStore(path)
    query = 'SECRET UNIQUE QUERY CONTENT'
    result = record(store, query, 100)
    store = MetricsStore(path)
    assert store.count() == 1
    row = store.recent()[0]
    assert row['request_id'] == result.request_id and row['status'] == 'SUCCESS'
    assert row['query_hash'] == sha256(query.encode('utf-8')).hexdigest()
    assert row['query_character_count'] == len(query)
    assert row['search_model'] == result.routing.search_model
    assert row['writer_model'] == row['analyst_model'] == result.routing.writer_model
    assert row['selected_route'] == 'fast' and row['requested_route_mode'] == 'fast'
    assert row['routing_ms'] == .1 and row['total_ms'] == 100
    assert 'query' not in row and 'final_answer' not in row
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == SCHEMA_VERSION == 1
        assert db.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
        dump = '\n'.join(db.iterdump())
    assert query not in dump and 'PRIVATE' not in dump
    assert query.encode() not in path.read_bytes()


def test_recent_summary_percentiles_and_cache_denominator(tmp_path):
    store = MetricsStore(tmp_path / 'metrics.db')
    assert store.count() == 0 and store.recent() == []
    assert store.summary()['p50_latency_ms'] is None
    assert store.summary()['cache_hit_rate'] is None
    record(store, 'first', 10)
    record(store, 'second', 20, cache='EXACT_HIT')
    record(store, 'third', 30, cache='SEMANTIC_HIT', mode='quality')
    record(store, 'fourth', 40, status='ERROR', mode='quality', rag=True)
    last = record(store, 'fifth', 50, cache='DISABLED')
    assert store.recent(limit=2)[0]['request_id'] == last.request_id
    summary = store.summary()
    assert summary == dict(total_requests=5, successful_requests=4, failed_requests=1,
                          cache_hits=2, cache_lookups=4, cache_hit_rate=.5,
                          fast_count=3, quality_count=2, average_latency_ms=30,
                          p50_latency_ms=30, p95_latency_ms=48,
                          ddgs_usage_count=2, ddgs_call_count=2, rag_usage_count=1)


def test_singleton_and_even_percentiles(tmp_path):
    store = MetricsStore(tmp_path / 'metrics.db')
    record(store, 'one', 10)
    assert store.summary()['p95_latency_ms'] == 10
    record(store, 'two', 30)
    assert store.summary()['p50_latency_ms'] == 20
    assert store.summary()['p95_latency_ms'] == 29


@pytest.mark.parametrize('limit', [0, -1, True, '1', 1.5])
def test_recent_rejects_invalid_limit(tmp_path, limit):
    with pytest.raises(ValueError):
        MetricsStore(tmp_path / 'metrics.db').recent(limit)


def test_exact_query_hash_does_not_change_cache_normalization(tmp_path):
    store = MetricsStore(tmp_path / 'metrics.db')
    record(store, 'What is MCP?', 1)
    record(store, 'What  is MCP?', 1)
    record(store, 'What is MCP?', 1)
    hashes = [row['query_hash'] for row in store.recent()]
    assert hashes[0] == hashes[2] and hashes[0] != hashes[1]


def test_independent_connections_allow_concurrent_writes(tmp_path):
    store = MetricsStore(tmp_path / 'metrics.db')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda index: record(store, str(index), index + 1), range(8)))
    assert store.count() == 8


def test_unknown_schema_is_rejected_without_overwrite(tmp_path):
    path = tmp_path / 'metrics.db'
    with sqlite3.connect(path) as db:
        db.execute('PRAGMA user_version=999')
    with pytest.raises(ValueError, match='Unsupported metrics schema'):
        MetricsStore(path)
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 999
