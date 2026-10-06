"""Precision-first exact/semantic lookup with bounded web-answer freshness."""

from functools import lru_cache
from math import isfinite
from pathlib import Path
import re
from time import time
from typing import Callable

from rag.embeddings import EmbeddingProvider, OllamaEmbeddingProvider, validate_embeddings
from semantic_cache.models import CacheEntry, CacheHit, CacheLookup, CacheScope, entry_id, normalize_query
from semantic_cache.store import ChromaCacheStore


DEFAULT_TTL_SECONDS = 3600.
DEFAULT_SIMILARITY_THRESHOLD = .97


def queries_compatible(first: str, second: str) -> bool:
    """Small guard, motivated by calibration; no model call or query rewriting."""
    groups = (
        {'advantage', 'advantages', 'benefit', 'benefits', 'pros'},
        {'disadvantage', 'disadvantages', 'drawback', 'drawbacks', 'cons', 'risk', 'risks'},
        {'enable', 'enabling', 'activate', 'install'},
        {'disable', 'disabling', 'deactivate', 'uninstall'},
        {'establish', 'establishment', 'establishes'},
        {'terminate', 'termination', 'terminates'},
        {'not', 'no', 'never', "don't", "doesn't", 'without'},
    )

    def signature(query):
        tokens = set(re.findall(r"[\w']+", query.casefold()))
        return tuple(bool(tokens & group) for group in groups), set(re.findall(r'\d+(?:\.\d+)*', query))

    return signature(first) == signature(second)


def cacheable_answer(answer: str) -> bool:
    if not isinstance(answer, str) or not answer.strip():
        return False
    text = answer.strip().casefold()
    return not text.startswith(('error:', 'error occurred', 'failed:', 'exception:', 'traceback '))


class SemanticCacheService:
    def __init__(self, embedding_provider: EmbeddingProvider | None = None,
                 store: ChromaCacheStore | None = None, *, ttl_seconds: float = DEFAULT_TTL_SECONDS,
                 similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                 clock: Callable[[], float] = time):
        if isinstance(ttl_seconds, bool) or not isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError('TTL must be finite and positive')
        if isinstance(similarity_threshold, bool) or not isfinite(similarity_threshold) or not 0 <= similarity_threshold <= 1:
            raise ValueError('Similarity threshold must be between 0 and 1')
        self.embedding_provider = embedding_provider if embedding_provider is not None else OllamaEmbeddingProvider()
        self.store = store if store is not None else ChromaCacheStore()
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.clock = clock

    def _live(self, entry: CacheEntry, scope: CacheScope, now: float) -> bool:
        return (entry.scope.digest == scope.digest and entry.created_at <= now
                < min(entry.expires_at, entry.created_at + self.ttl_seconds))

    def lookup(self, query: str, scope: CacheScope) -> CacheLookup:
        identity = entry_id(query, scope)
        now = self.clock()
        exact = self.store.get_exact(identity)
        if exact is not None and self._live(exact, scope, now):
            return CacheLookup(CacheHit(exact, 'exact', 1.0))
        if not self.store.has_live_entries(scope, now, self.ttl_seconds):
            return CacheLookup()
        embedding = tuple(self.embedding_provider.embed_text(query))
        validate_embeddings([list(embedding)], 1)
        if not any(embedding):
            raise ValueError('Cache embedding must be nonzero')
        candidate = self.store.search(embedding, scope, now, self.ttl_seconds)
        if candidate is not None:
            entry, similarity = candidate
            if (isfinite(similarity) and similarity > self.similarity_threshold
                    and self._live(entry, scope, self.clock()) and queries_compatible(query, entry.query)):
                return CacheLookup(CacheHit(entry, 'semantic', similarity), embedding)
        return CacheLookup(embedding=embedding)

    def save(self, query: str, answer: str, scope: CacheScope,
             *, embedding: tuple[float, ...] | None = None) -> bool:
        if not cacheable_answer(answer):
            return False
        normalized = normalize_query(query)
        vector = embedding if embedding is not None else tuple(self.embedding_provider.embed_text(query))
        now = self.clock()
        self.store.upsert(CacheEntry(entry_id(query, scope), query, normalized, answer,
                                     now, now + self.ttl_seconds, tuple(vector), scope))
        return True


@lru_cache(maxsize=1)
def get_default_cache_service() -> SemanticCacheService:
    """Lazy local persistence; construction never calls the embedding model."""
    return SemanticCacheService(store=ChromaCacheStore(
        Path(__file__).resolve().parents[1] / 'data' / 'semantic_cache',
    ))
