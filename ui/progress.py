"""Native Streamlit progress rendering, separate from the final console."""

import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

from observability.progress import ExecutionProgressEvent, ExecutionStage, StageStatus


STAGE_LABELS = {
    ExecutionStage.ROUTING: 'Routing', ExecutionStage.CACHE: 'Cache',
    ExecutionStage.RAG: 'RAG', ExecutionStage.WEB_SEARCH: 'Web Search',
    ExecutionStage.WEB_SEARCHER: 'Web Searcher', ExecutionStage.ANALYST: 'Analyst',
    ExecutionStage.WRITER: 'Writer', ExecutionStage.FINALIZING: 'Finalizing',
    ExecutionStage.COMPLETE: 'Complete',
}
SYMBOLS = {StageStatus.STARTED: '●', StageStatus.COMPLETED: '✓',
           StageStatus.SKIPPED: '–', StageStatus.ERROR: '!'}


def format_stage(stage, event=None):
    if event is None:
        return f'○ {STAGE_LABELS[stage]} — pending'
    duration = '' if event.elapsed_ms is None else f' · {event.elapsed_ms / 1000:.2f} s'
    return f'{SYMBOLS[event.status]} {STAGE_LABELS[stage]} — {event.message}{duration}'


class ResearchProgress:
    """Transient state only. All rendering happens on the submitting script run."""

    def __init__(self):
        self.events = {}
        self.request_id = None
        self.had_warning = False
        self.web_failed = False
        self.status = st.status('Researching...', expanded=True, state='running')
        self.rows = {stage: self.status.empty() for stage in STAGE_LABELS}
        for stage, row in self.rows.items():
            row.text(format_stage(stage))

    def accept(self, event: ExecutionProgressEvent):
        if self.request_id is None:
            self.request_id = event.request_id
        if event.request_id != self.request_id:
            return
        self.events[event.stage] = event
        text = format_stage(event.stage, event)
        # Keep failures visible even when the Searcher makes a later successful call.
        if event.stage == ExecutionStage.WEB_SEARCH:
            if event.status == StageStatus.ERROR:
                self.web_failed = True
            elif self.web_failed:
                text += ' · earlier search failed'
        if event.status == StageStatus.ERROR:
            self.had_warning = True
        self.rows[event.stage].text(text)
        if event.stage == ExecutionStage.COMPLETE:
            self.finish(event.status == StageStatus.COMPLETED)

    def finish(self, success):
        label = 'Research complete' if success else 'Research failed'
        if success and self.had_warning:
            label += ' (with warnings)'
        self.status.update(label=label, state='complete' if success else 'error', expanded=True)

    def run(self, operation, *args, **kwargs):
        """Invoke the canonical backend once; render queued events on this thread.

        CrewAI may run tool calls on additional threads. The sink only enqueues;
        no worker touches Streamlit or session state. Waiting is for actual work,
        with no artificial progress delays or token streaming.
        """
        events = Queue()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='research') as worker:
            future = worker.submit(operation, *args, progress_callback=events.put, **kwargs)
            while not future.done():
                try:
                    self.accept(events.get(timeout=0.1))
                except Empty:
                    pass
            # Completion can race the last callback; render all remaining events.
            while True:
                try:
                    self.accept(events.get_nowait())
                except Empty:
                    break
            return future.result()
