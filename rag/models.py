"""Typed document records with immutable scalar metadata."""

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Mapping


MetadataValue = str | int | float | bool
Metadata = Mapping[str, MetadataValue]


def freeze_metadata(metadata: Metadata) -> Metadata:
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, (str, int, float, bool)):
            raise ValueError("Metadata must have string keys and scalar values")
        if isinstance(value, float) and not isfinite(value):
            raise ValueError("Metadata numbers must be finite")
    return MappingProxyType(dict(metadata))


@dataclass(frozen=True)
class DocumentPage:
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Document:
    source: str
    pages: tuple[DocumentPage, ...]
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "pages", tuple(self.pages))
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    source: str
    page: int | None
    chunk_index: int
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    page: int | None
    chunk_index: int
    distance: float | None = None
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
