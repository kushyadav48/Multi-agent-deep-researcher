"""Transient, request-local stage events; never prompts or model output."""

from collections.abc import Callable
from enum import Enum
from threading import RLock
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field, field_validator

from observability.recorder import utc_now


class ExecutionStage(str, Enum):
    ROUTING = 'ROUTING'
    CACHE = 'CACHE'
    RAG = 'RAG'
    WEB_SEARCH = 'WEB_SEARCH'
    WEB_SEARCHER = 'WEB_SEARCHER'
    ANALYST = 'ANALYST'
    WRITER = 'WRITER'
    FINALIZING = 'FINALIZING'
    COMPLETE = 'COMPLETE'


class StageStatus(str, Enum):
    STARTED = 'STARTED'
    COMPLETED = 'COMPLETED'
    SKIPPED = 'SKIPPED'
    ERROR = 'ERROR'


class ExecutionProgressEvent(BaseModel):
    """Explicit metadata allowlist. Producers must use safe observable messages."""

    model_config = ConfigDict(extra='forbid', frozen=True, allow_inf_nan=False)

    request_id: str
    stage: ExecutionStage
    status: StageStatus
    message: str
    timestamp: str = Field(default_factory=utc_now)
    elapsed_ms: float | None = Field(default=None, ge=0)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator('metadata', mode='before')
    @classmethod
    def safe_metadata(cls, value):
        allowed = {'route', 'complexity', 'score', 'threshold', 'cache_status',
                   'retrieval_status', 'chunk_count', 'query', 'result_count',
                   'error_type', 'reason'}
        if not isinstance(value, dict) or value.keys() - allowed:
            raise ValueError('Unsupported progress metadata field')
        if any(type(item) not in (str, int, float, bool, type(None)) for item in value.values()):
            raise ValueError('Progress metadata must contain only scalar values')
        return value


ProgressCallback = Callable[[ExecutionProgressEvent], None]
AGENT_STAGES = (ExecutionStage.WEB_SEARCHER, ExecutionStage.ANALYST, ExecutionStage.WRITER)


class ProgressEmitter:
    """One safe sink per request. A broken sink is disabled after one warning."""

    def __init__(self, result, callback: ProgressCallback | None):
        self.result = result
        self.callback = callback
        self.active = ExecutionStage.ROUTING
        self.finalizing_started = None
        self.completed_agents = set()
        self.web_invoked = False
        self._lock = RLock()

    def emit(self, stage, status, message, *, elapsed_ms=None, metadata=None):
        # CrewAI can invoke several tools concurrently. Serialize sink access and
        # disable a failing sink exactly once without affecting those tool calls.
        with self._lock:
            if self.callback is None:
                return
            try:
                self.callback(ExecutionProgressEvent(
                    request_id=self.result.request_id, stage=stage, status=status,
                    message=message, elapsed_ms=elapsed_ms, metadata=metadata or {},
                ))
            except Exception as error:
                self.result.warnings.append(f'Progress callback failed ({type(error).__name__}).')
                self.callback = None

    def start(self, stage, message='running...'):
        self.active = stage
        self.emit(stage, StageStatus.STARTED, message)

    def error(self, stage, error_type, message='failed'):
        self.emit(stage, StageStatus.ERROR, message, metadata={'error_type': error_type})

    def finalizing(self):
        if self.finalizing_started is None:
            self.finalizing_started = perf_counter()
            self.start(ExecutionStage.FINALIZING)

    def finish(self):
        if self.result.status == 'SUCCESS':
            self.emit(ExecutionStage.FINALIZING, StageStatus.COMPLETED, 'completed',
                      elapsed_ms=(perf_counter() - self.finalizing_started) * 1000)
            self.emit(ExecutionStage.COMPLETE, StageStatus.COMPLETED, 'Research complete',
                      elapsed_ms=self.result.timings.total_ms)
        else:
            self.error(ExecutionStage.COMPLETE, self.result.error_type or 'ResearchError',
                       'Research failed')

    def agent_completed(self, index):
        stage = AGENT_STAGES[index]
        if stage in self.completed_agents:
            return
        self.completed_agents.add(stage)
        self.emit(stage, StageStatus.COMPLETED, 'completed')
        if index + 1 < len(AGENT_STAGES):
            self.start(AGENT_STAGES[index + 1])

    def attach_crew(self, crew):
        """Compose public Task.callback; CrewAI invokes Crew.task_callback itself.

        Return existing awaitables so CrewAI retains its synchronous/async callback
        behavior. A callback shared by Task and Crew must still execute only once.
        """
        if self.callback is None:
            return
        for index, task in enumerate(crew.tasks):
            previous = task.callback
            if previous == crew.task_callback:
                previous = None

            def completed(output, index=index, previous=previous):
                self.agent_completed(index)
                return previous(output) if previous is not None else None

            task.callback = completed

    def public_outputs(self):
        """Reconcile public outputs after kickoff, including deterministic fakes."""
        if not self.web_invoked:
            self.emit(ExecutionStage.WEB_SEARCH, StageStatus.SKIPPED, 'not invoked')
        for index, agent in enumerate(self.result.agents):
            if agent.status == 'COMPLETED':
                self.agent_completed(index)
            elif AGENT_STAGES[index] not in self.completed_agents:
                self.emit(AGENT_STAGES[index], StageStatus.SKIPPED, 'output unavailable')
