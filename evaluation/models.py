"""Compact benchmark records: no prompts, messages, answers, or vectors."""

from dataclasses import asdict, dataclass, field

from evaluation import BENCHMARK_VERSION


@dataclass
class BenchmarkCaseResult:
    case_id: str
    suite: str
    status: str
    duration_ms: float | None = None
    route: str | None = None
    models: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    tokens: dict = field(default_factory=dict)
    measured: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class BenchmarkSuiteResult:
    name: str
    category: str
    cases: list[BenchmarkCaseResult] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


@dataclass
class BenchmarkRun:
    timestamp: str
    git_commit: str
    environment: dict
    configuration: dict
    source_hash: str
    benchmark_version: str = BENCHMARK_VERSION
    research_invocations: int = 0
    research_reservations: int = 0
    crew_executions: int = 0
    crew_reservations: int = 0
    suites: list[BenchmarkSuiteResult] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        values["suites"] = [BenchmarkSuiteResult(
            name=suite["name"], category=suite["category"], metrics=suite["metrics"],
            cases=[BenchmarkCaseResult(**case) for case in suite["cases"]],
        ) for suite in data["suites"]]
        return cls(**values)
