"""Separate embedded Chroma collection; never uses automatic embeddings."""

from dataclasses import asdict
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings

from semantic_cache.models import CacheEntry, CacheScope


class ChromaCacheStore:
    def __init__(self, path: str | Path = 'data/semantic_cache'):
        self.path = Path(path).expanduser().resolve()
        self._client = chromadb.PersistentClient(
            path=str(self.path), settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name='deep_researcher_semantic_cache', embedding_function=None,
            configuration={'hnsw': {'space': 'cosine'}},
        )

    def count(self) -> int:
        return self._collection.count()

    def upsert(self, entry: CacheEntry) -> None:
        self._collection.upsert(
            ids=[entry.id], documents=[entry.answer], embeddings=[list(entry.embedding)],
            metadatas=[dict(query=entry.query, normalized_query=entry.normalized_query,
                            created_at=entry.created_at, expires_at=entry.expires_at,
                            scope=entry.scope.digest, scope_json=json.dumps(asdict(entry.scope)))],
        )

    @staticmethod
    def _entry(identity, answer, metadata, embedding) -> CacheEntry:
        scope = CacheScope(**json.loads(metadata['scope_json']))
        if metadata['scope'] != scope.digest:
            raise ValueError('Cache scope metadata is inconsistent')
        return CacheEntry(identity, metadata['query'], metadata['normalized_query'], answer,
                          metadata['created_at'], metadata['expires_at'], tuple(embedding), scope)

    def get_exact(self, identity: str) -> CacheEntry | None:
        result = self._collection.get(ids=[identity], include=['documents', 'metadatas', 'embeddings'])
        if not result['ids']:
            return None
        return self._entry(result['ids'][0], result['documents'][0],
                           result['metadatas'][0], result['embeddings'][0])

    @staticmethod
    def _where(scope: CacheScope, now: float, ttl: float):
        return {'$and': [{'scope': scope.digest}, {'expires_at': {'$gt': now}},
                         {'created_at': {'$gt': now - ttl}}]}

    def has_live_entries(self, scope: CacheScope, now: float, ttl: float) -> bool:
        if self.count() == 0:
            return False
        return bool(self._collection.get(where=self._where(scope, now, ttl), limit=1, include=[])['ids'])

    def search(self, embedding: tuple[float, ...], scope: CacheScope,
               now: float, ttl: float) -> tuple[CacheEntry, float] | None:
        result = self._collection.query(
            query_embeddings=[list(embedding)], n_results=1, where=self._where(scope, now, ttl),
            include=['documents', 'metadatas', 'embeddings', 'distances'],
        )
        if not result['ids'][0]:
            return None
        entry = self._entry(result['ids'][0][0], result['documents'][0][0],
                            result['metadatas'][0][0], result['embeddings'][0][0])
        return entry, 1.0 - result['distances'][0][0]
