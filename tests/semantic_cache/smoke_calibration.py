"""Opt-in local threshold calibration: python -m tests.semantic_cache.smoke_calibration."""

import json
from math import sqrt
from time import perf_counter

from rag.embeddings import OllamaEmbeddingProvider

PAIRS = [
    ('positive', 'What is the Model Context Protocol?', 'Explain the Model Context Protocol.'),
    ('positive', 'How does RAG work?', 'Explain retrieval augmented generation.'),
    ('positive', 'What is photosynthesis?', 'Explain photosynthesis.'),
    ('positive', 'How does TCP establish a connection?', 'Explain how TCP establishes a connection.'),
    ('negative', 'What is MCP?', 'How does photosynthesis work?'),
    ('negative', 'What is the Model Context Protocol?', 'What is the Transmission Control Protocol?'),
    ('negative', 'How does RAG work?', 'How do I bake sourdough bread?'),
    ('hard_negative', 'What are the advantages of MCP?', 'What are the disadvantages of MCP?'),
    ('hard_negative', 'How do I enable caching in Python?', 'How do I disable caching in Python?'),
    ('hard_negative', 'Explain TCP connection establishment.', 'Explain TCP connection termination.'),
    ('hard_negative', 'What is the Model Context Protocol?', 'What are the security risks of the Model Context Protocol?'),
]

provider = OllamaEmbeddingProvider()
started = perf_counter()
provider.embed_text(PAIRS[0][1])
first_seconds = perf_counter() - started
vectors = provider.embed_texts([s for _, a, b in PAIRS for s in (a, b)])
results = []
for i, (kind, a, b) in enumerate(PAIRS):
    x, y = vectors[2*i:2*i+2]
    similarity = sum(u*v for u,v in zip(x,y)) / sqrt(sum(u*u for u in x)*sum(v*v for v in y))
    results.append(dict(kind=kind, query=a, candidate=b, similarity=similarity))
print(json.dumps(dict(dimension=len(vectors[0]), first_embedding_seconds=first_seconds, pairs=results), indent=2))
