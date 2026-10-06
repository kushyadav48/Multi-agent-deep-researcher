import json
import subprocess
import sys

from semantic_cache.models import CacheScope
from semantic_cache.service import SemanticCacheService
from semantic_cache.store import ChromaCacheStore
from tests.rag.helpers import FakeEmbeddingProvider


def test_persistence_in_recreated_service_and_process(tmp_path):
    path = tmp_path / 'cache'
    scope = CacheScope(use_rag=False)
    cache = SemanticCacheService(FakeEmbeddingProvider(), ChromaCacheStore(path))
    cache.save('protocol tools', 'persisted answer', scope)
    recreated = SemanticCacheService(FakeEmbeddingProvider(), ChromaCacheStore(path))
    assert recreated.lookup('protocol tools', scope).hit.entry.answer == 'persisted answer'
    script = (
        'import json,sys; from semantic_cache.models import CacheScope; '
        'from semantic_cache.service import SemanticCacheService; '
        'from semantic_cache.store import ChromaCacheStore; '
        'from tests.rag.helpers import FakeEmbeddingProvider; '
        's=SemanticCacheService(FakeEmbeddingProvider(),ChromaCacheStore(sys.argv[1])); '
        'print(json.dumps(s.lookup("protocol tools",CacheScope(use_rag=False)).hit.entry.answer))'
    )
    result = subprocess.check_output([sys.executable, '-c', script, str(path)], text=True)
    assert json.loads(result) == 'persisted answer'
