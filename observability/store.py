"""Small SQLite operational history. Research content is never serialized here."""

from contextlib import contextmanager
from hashlib import sha256
from functools import lru_cache
from math import floor
from pathlib import Path
import sqlite3


SCHEMA_VERSION = 1
DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / 'data' / 'metrics' / 'research_metrics.db'


@lru_cache(maxsize=1)
def get_default_metrics_store():
    """Lazy service only; no connection is retained between operations."""
    return MetricsStore()

# Deliberate allowlist, independent of the rich trace schema.
_COLUMNS = {
    'request_id': 'TEXT PRIMARY KEY', 'started_at_utc': 'TEXT NOT NULL',
    'finished_at_utc': 'TEXT NOT NULL', 'status': 'TEXT NOT NULL',
    'query_hash': 'TEXT NOT NULL', 'query_character_count': 'INTEGER NOT NULL',
    'requested_route_mode': 'TEXT', 'selected_route': 'TEXT', 'routing_score': 'INTEGER',
    'search_model': 'TEXT', 'analyst_model': 'TEXT', 'writer_model': 'TEXT',
    'cache_status': 'TEXT', 'cache_similarity': 'REAL',
    'rag_enabled': 'INTEGER', 'rag_used': 'INTEGER', 'rag_chunk_count': 'INTEGER',
    'ddgs_call_count': 'INTEGER', 'web_result_count': 'INTEGER',
    'routing_ms': 'REAL', 'cache_lookup_ms': 'REAL', 'rag_retrieval_ms': 'REAL',
    'web_search_ms': 'REAL', 'crew_ms': 'REAL', 'total_ms': 'REAL',
    'input_tokens': 'INTEGER', 'output_tokens': 'INTEGER', 'total_tokens': 'INTEGER',
    'error_type': 'TEXT',
}


def _percentile(values, fraction):
    """Linear interpolation at (n - 1) * fraction, including singleton lists."""
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = floor(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


class MetricsStore:
    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise ValueError(f'Unsupported metrics schema version: {version}')
            connection.execute('PRAGMA journal_mode=WAL')
            with connection:
                fields = ', '.join(f'{key} {kind}' for key, kind in _COLUMNS.items())
                connection.execute(f'CREATE TABLE IF NOT EXISTS requests ({fields})')
                connection.execute('CREATE INDEX IF NOT EXISTS requests_started ON requests(started_at_utc)')
                connection.execute(f'PRAGMA user_version={SCHEMA_VERSION}')

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def record(self, result):
        route = result.routing
        row = dict(
            request_id=result.request_id, started_at_utc=result.started_at,
            finished_at_utc=result.finished_at, status=result.status,
            query_hash=sha256(result.query.encode('utf-8')).hexdigest(),
            query_character_count=len(result.query), requested_route_mode=result.requested_route_mode,
            selected_route=route.selected_route if route else None,
            routing_score=route.score if route else None,
            search_model=route.search_model if route else None,
            analyst_model=route.analyst_model if route else None,
            writer_model=route.writer_model if route else None,
            cache_status=result.cache.status, cache_similarity=result.cache.similarity,
            rag_enabled=int(result.rag.enabled), rag_used=int(result.rag.used),
            rag_chunk_count=result.rag.chunk_count, ddgs_call_count=result.web.total_calls,
            web_result_count=result.web.total_results, error_type=result.error_type,
        )
        for key in ('routing_ms', 'cache_lookup_ms', 'rag_retrieval_ms', 'web_search_ms', 'crew_ms', 'total_ms'):
            row[key] = getattr(result.timings, key)
        for key in ('input_tokens', 'output_tokens', 'total_tokens'):
            row[key] = getattr(result.usage, key)
        keys = ', '.join(_COLUMNS)
        placeholders = ', '.join('?' for _ in _COLUMNS)
        with self._connection() as connection, connection:
            connection.execute(f'INSERT INTO requests ({keys}) VALUES ({placeholders})',
                               [row[key] for key in _COLUMNS])

    def count(self) -> int:
        with self._connection() as connection:
            return connection.execute('SELECT COUNT(*) FROM requests').fetchone()[0]

    def recent(self, limit: int = 20) -> list[dict]:
        if type(limit) is not int or limit <= 0:
            raise ValueError('limit must be a positive integer')
        with self._connection() as connection:
            return [dict(row) for row in connection.execute(
                'SELECT * FROM requests ORDER BY started_at_utc DESC, rowid DESC LIMIT ?', (limit,),
            )]

    def summary(self) -> dict:
        with self._connection() as connection:
            rows = connection.execute(
                'SELECT status, cache_status, selected_route, total_ms, ddgs_call_count, rag_used FROM requests',
            ).fetchall()
        total = len(rows)
        hits = sum(row['cache_status'] in ('EXACT_HIT', 'SEMANTIC_HIT') for row in rows)
        eligible = sum(row['cache_status'] in ('EMPTY', 'MISS', 'EXACT_HIT', 'SEMANTIC_HIT') for row in rows)
        latencies = sorted(row['total_ms'] for row in rows if row['total_ms'] is not None)
        return dict(
            total_requests=total, successful_requests=sum(row['status'] == 'SUCCESS' for row in rows),
            failed_requests=sum(row['status'] == 'ERROR' for row in rows), cache_hits=hits,
            cache_lookups=eligible, cache_hit_rate=hits / eligible if eligible else None,
            fast_count=sum(row['selected_route'] == 'fast' for row in rows),
            quality_count=sum(row['selected_route'] == 'quality' for row in rows),
            average_latency_ms=sum(latencies) / len(latencies) if latencies else None,
            p50_latency_ms=_percentile(latencies, .5), p95_latency_ms=_percentile(latencies, .95),
            ddgs_usage_count=sum(row['ddgs_call_count'] > 0 for row in rows),
            ddgs_call_count=sum(row['ddgs_call_count'] for row in rows),
            rag_usage_count=sum(row['rag_used'] for row in rows),
        )
