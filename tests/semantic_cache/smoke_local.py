"""Opt-in embedding-only real cache check: python -m tests.semantic_cache.smoke_local."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

from rag.embeddings import OllamaEmbeddingProvider
from semantic_cache.models import CacheScope
from semantic_cache.service import DEFAULT_SIMILARITY_THRESHOLD, SemanticCacheService
from semantic_cache.store import ChromaCacheStore


QUERY = 'What is the Model Context Protocol?'
PARAPHRASE = 'Explain the Model Context Protocol.'


def run_smoke(path):
    cache = SemanticCacheService(OllamaEmbeddingProvider(), ChromaCacheStore(path))
    scope = CacheScope(use_rag=False)
    started = perf_counter()
    cache.save(QUERY, 'MCP connects AI applications to external tools and context.', scope)
    insertion_seconds = perf_counter() - started
    started = perf_counter()
    lookup = cache.lookup(PARAPHRASE, scope)
    lookup_seconds = perf_counter() - started
    assert lookup.hit is not None and lookup.hit.kind == 'semantic'
    assert lookup.hit.entry.answer == 'MCP connects AI applications to external tools and context.'
    assert cache.lookup('How does photosynthesis work?', scope).hit is None
    hard_scope = CacheScope(use_rag=False, version='hard-negative-smoke')
    cache.save('What are the advantages of MCP?', 'Advantages answer.', hard_scope)
    assert cache.lookup('What are the disadvantages of MCP?', hard_scope).hit is None
    print(json.dumps(dict(dimension=len(lookup.embedding), threshold=DEFAULT_SIMILARITY_THRESHOLD,
                         semantic_hit=True, similarity=lookup.hit.similarity,
                         distance=1-lookup.hit.similarity, insertion_seconds=insertion_seconds,
                         warm_lookup_seconds=lookup_seconds, unrelated_miss=True,
                         hard_negative_miss=True), indent=2))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        run_smoke(Path(sys.argv[2]))
    else:
        # Child process releases Chroma's Windows file handles before cleanup.
        with tempfile.TemporaryDirectory(prefix='research_cache_local_') as directory:
            subprocess.run([sys.executable, '-m', 'tests.semantic_cache.smoke_local',
                            '--worker', directory], check=True)
