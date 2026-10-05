"""Deterministic character windows that never cross PDF page boundaries."""

from dataclasses import dataclass
from hashlib import sha256
import json

from rag.models import Document, DocumentChunk


@dataclass(frozen=True)
class TextChunker:
    chunk_size: int = 1000
    overlap: int = 200

    def __post_init__(self):
        if type(self.chunk_size) is not int or self.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if type(self.overlap) is not int or not 0 <= self.overlap < self.chunk_size:
            raise ValueError("overlap must be an integer >= 0 and < chunk_size")

    def chunk(self, document: Document) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        step = self.chunk_size - self.overlap
        for page in document.pages:
            for start in range(0, len(page.text), step):
                text = page.text[start:start + self.chunk_size]
                if text.strip():
                    identity = json.dumps(
                        [document.source, page.page, start, self.chunk_size, self.overlap, text],
                        ensure_ascii=False, separators=(",", ":"),
                    )
                    chunks.append(DocumentChunk(
                        id=sha256(identity.encode("utf-8")).hexdigest(),
                        text=text, source=document.source, page=page.page,
                        chunk_index=len(chunks), metadata=document.metadata,
                    ))
                if start + self.chunk_size >= len(page.text):
                    break
        return chunks
