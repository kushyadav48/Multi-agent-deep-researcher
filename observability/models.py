"""Explicit output allowlist: never copy prompts, messages, or embeddings."""

from dataclasses import asdict, dataclass, field
from typing import Literal


CacheStatus = Literal['DISABLED', 'EMPTY', 'MISS', 'EXACT_HIT', 'SEMANTIC_HIT', 'ERROR', 'BYPASSED']
RAGStatus = Literal['DISABLED', 'EMPTY_CORPUS', 'NO_RELEVANT_RESULTS', 'USED', 'ERROR', 'BYPASSED', 'NOT_EXECUTED']
AgentStatus = Literal['NOT_EXECUTED', 'COMPLETED', 'ERROR', 'UNAVAILABLE']


@dataclass
class RoutingTrace:
    requested_mode: str
    selected_route: str
    complexity: Literal['SIMPLE', 'COMPLEX']
    score: int
    threshold: int
    reasons: tuple[str, ...]
    policy_version: str
    search_model: str
    analyst_model: str
    writer_model: str


@dataclass
class CacheTrace:
    status: CacheStatus = 'DISABLED'
    similarity: float | None = None
    distance: float | None = None
    threshold: float | None = None
    entry_id: str | None = None
    reason: str | None = None


@dataclass
class RAGEvidence:
    source: str
    page: int | None
    chunk_index: int
    distance: float | None
    text: str


@dataclass
class RAGTrace:
    enabled: bool = False
    status: RAGStatus = 'DISABLED'
    evidence: list[RAGEvidence] = field(default_factory=list)
    duration_ms: float | None = None
    reason: str | None = None

    @property
    def used(self) -> bool:
        return self.status == 'USED'

    @property
    def chunk_count(self) -> int:
        return len(self.evidence)


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class WebSearchCall:
    query: str
    results: list[WebSearchResult]
    duration_ms: float
    status: Literal['SUCCESS', 'EMPTY', 'ERROR']
    error_type: str | None = None


@dataclass
class WebTrace:
    calls: list[WebSearchCall] = field(default_factory=list)

    @property
    def used(self) -> bool:
        return bool(self.calls)

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    @property
    def total_results(self) -> int:
        return sum(len(call.results) for call in self.calls)


@dataclass
class AgentOutput:
    agent_name: str
    task_name: str
    model: str | None = None
    status: AgentStatus = 'NOT_EXECUTED'
    output_text: str | None = None
    reason: str | None = 'not_started'


@dataclass
class Timings:
    """Milliseconds from perf_counter; None means that stage did not execute.

    Web time sums real tool calls and is included in crew time. Total includes
    routing, setup, retrieval, Crew, and cache writes, but excludes metrics I/O.
    """
    routing_ms: float | None = None
    cache_lookup_ms: float | None = None
    rag_retrieval_ms: float | None = None
    web_search_ms: float | None = None
    crew_ms: float | None = None
    total_ms: float | None = None


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class ResearchExecutionResult:
    request_id: str
    query: str
    started_at: str
    requested_route_mode: str
    status: Literal['SUCCESS', 'ERROR'] = 'SUCCESS'
    finished_at: str | None = None
    routing: RoutingTrace | None = None
    cache: CacheTrace = field(default_factory=CacheTrace)
    rag: RAGTrace = field(default_factory=RAGTrace)
    web: WebTrace = field(default_factory=WebTrace)
    agents: list[AgentOutput] = field(default_factory=lambda: [
        AgentOutput('Web Searcher', 'web_search'),
        AgentOutput('Research Analyst', 'analysis'),
        AgentOutput('Technical Writer', 'writing'),
    ])
    timings: Timings = field(default_factory=Timings)
    usage: Usage = field(default_factory=Usage)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    error_type: str | None = None
    final_answer: str = ''
    metrics_persisted: bool = False

    def to_dict(self) -> dict:
        """JSON-ready trace including the derived UI counters."""
        result = asdict(self)
        result['rag'].update(used=self.rag.used, chunk_count=self.rag.chunk_count)
        result['web'].update(used=self.web.used, total_calls=self.web.total_calls,
                             total_results=self.web.total_results)
        return result
