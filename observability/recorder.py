"""Bounded observation of public execution outputs, scoped to one request."""

from datetime import datetime, timezone
from contextlib import contextmanager
from time import perf_counter
from uuid import uuid4

from observability.models import RAGEvidence, ResearchExecutionResult, RoutingTrace, Usage
from rag.context import source_name
from routing.router import QUALITY_THRESHOLD


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExecutionRecorder:
    def __init__(self, query, *, use_rag, use_cache, model_route):
        self.started = perf_counter()
        mode = model_route.value if hasattr(model_route, 'value') else str(model_route)
        self.result = ResearchExecutionResult(str(uuid4()), query, utc_now(), mode)
        self.result.rag.enabled = use_rag
        self.result.rag.status = 'NOT_EXECUTED' if use_rag else 'DISABLED'
        self.result.cache.status = 'BYPASSED' if use_cache else 'DISABLED'

    @contextmanager
    def measure(self, stage):
        started = perf_counter()
        try:
            yield
        finally:
            elapsed = (perf_counter() - started) * 1000
            setattr(self.result.timings, stage, elapsed)
            if stage == 'rag_retrieval_ms':
                self.result.rag.duration_ms = elapsed

    def routing(self, decision):
        self.result.routing = RoutingTrace(
            decision.requested_mode.value, decision.selected_route.value,
            'SIMPLE' if decision.selected_route.value == 'fast' else 'COMPLEX',
            decision.score, QUALITY_THRESHOLD, decision.reasons, decision.policy_version,
            decision.search_model, decision.synthesis_model, decision.synthesis_model,
        )
        self.result.requested_route_mode = decision.requested_mode.value
        for agent, model in zip(self.result.agents, (
            decision.search_model, decision.synthesis_model, decision.synthesis_model,
        )):
            agent.model = model

    def rag(self, status, chunks):
        self.result.rag.status = status
        self.result.rag.evidence = [RAGEvidence(
            source_name(chunk.source), chunk.page, chunk.chunk_index, chunk.distance, chunk.text,
        ) for chunk in chunks]

    def cache_hit(self):
        if self.result.rag.enabled:
            self.result.rag.status = 'BYPASSED'
            self.result.rag.reason = 'semantic_cache_hit'
        for agent in self.result.agents:
            agent.reason = 'semantic_cache_hit'

    def crew_outputs(self, output):
        # TaskOutput.messages/description/summary can contain private prompts.
        # Only raw (the returned task deliverable) is read, in known task order.
        tasks = getattr(output, 'tasks_output', None)
        if not isinstance(tasks, (list, tuple)) or len(tasks) != 3:
            self.result.warnings.append('Expected three public Crew task outputs; some outputs are unavailable.')
        for index, agent in enumerate(self.result.agents):
            task = tasks[index] if isinstance(tasks, (list, tuple)) and index < len(tasks) else None
            raw = getattr(task, 'raw', None)
            agent.output_text = raw if isinstance(raw, str) else None
            agent.status = 'COMPLETED' if isinstance(raw, str) and raw.strip() else 'UNAVAILABLE'
            agent.reason = None if agent.status == 'COMPLETED' else 'task_output_unavailable'
        usage = getattr(output, 'token_usage', None)
        values = [getattr(usage, key, None) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')]
        # CrewAI defaults unknown usage to zeros; do not present that as measured.
        if all(type(value) is int and value >= 0 for value in values):
            if values[2] > 0 and values[0] + values[1] == values[2]:
                self.result.usage = Usage(*values)

    def web_calls(self, calls):
        self.result.web.calls.extend(calls)
        if calls:
            self.result.timings.web_search_ms = sum(call.duration_ms for call in calls)

    def finish(self, metrics_store=None, *, record_metrics=True):
        self.result.finished_at = utc_now()
        self.result.timings.total_ms = (perf_counter() - self.started) * 1000
        if record_metrics:
            try:
                from observability.store import get_default_metrics_store
                store = metrics_store if metrics_store is not None else get_default_metrics_store()
                store.record(self.result)
                self.result.metrics_persisted = True
            except Exception as error:
                # Never print, nor copy exception content (which could include secrets).
                self.result.warnings.append(f'Metrics persistence failed ({type(error).__name__}).')
        return self.result
