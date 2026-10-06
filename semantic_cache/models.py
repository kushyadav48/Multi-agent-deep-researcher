"""Typed cache records and deterministic identities."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Literal

from rag.embeddings import validate_embeddings


CACHE_VERSION = '1'  # Bump when prompts, answer generation, or safety policy changes.
RESEARCH_MODEL = 'ollama/qwen2.5:3b'
EMBEDDING_MODEL = 'qwen3-embedding:0.6b'


def digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def normalize_query(query: str) -> str:
    """Collapse whitespace only; preserve case, punctuation, and identifiers."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Cache query must be non-empty text')
    return ' '.join(query.split())


@dataclass(frozen=True)
class CacheScope:
    research_model: str = RESEARCH_MODEL
    version: str = CACHE_VERSION
    embedding_model: str = EMBEDDING_MODEL
    rag_embedding_model: str = EMBEDDING_MODEL
    use_rag: bool = True
    corpus_fingerprint: str = 'EMPTY'
    rag_top_k: int = 4
    rag_max_distance: float = .6
    max_context_chars: int = 6000
    max_chunk_chars: int = 1000

    @property
    def digest(self) -> str:
        state = asdict(self)
        if not self.use_rag:
            state['corpus_fingerprint'] = 'WEB_ONLY'
            for key in ('rag_top_k', 'rag_max_distance', 'max_context_chars', 'max_chunk_chars', 'rag_embedding_model'):
                state.pop(key)
        return digest(state)


def entry_id(query: str, scope: CacheScope) -> str:
    return digest([scope.digest, normalize_query(query)])


@dataclass(frozen=True)
class CacheEntry:
    id: str
    query: str
    normalized_query: str
    answer: str
    created_at: float
    expires_at: float
    embedding: tuple[float, ...]
    scope: CacheScope

    def __post_init__(self):
        validate_embeddings([list(self.embedding)], 1)
        if not any(self.embedding):
            raise ValueError('Cache embedding must be nonzero')
        if self.id != entry_id(self.query, self.scope) or self.normalized_query != normalize_query(self.query):
            raise ValueError('Cache record identity does not match its query/scope')
        if not self.answer.strip() or not 0 <= self.created_at < self.expires_at < float('inf'):
            raise ValueError('Cache answer and timestamps must be valid')


@dataclass(frozen=True)
class CacheHit:
    entry: CacheEntry
    kind: Literal['exact', 'semantic']
    similarity: float


@dataclass(frozen=True)
class CacheLookup:
    hit: CacheHit | None = None
    embedding: tuple[float, ...] | None = None
