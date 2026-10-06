"""Opt-in bounded two-call live regression; never discovered by pytest."""

from contextlib import redirect_stdout
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

import agents
from crewai.events import crewai_event_bus
from ddgs.ddgs import DDGS as ConcreteDDGS
from rag.embeddings import OllamaEmbeddingProvider
from semantic_cache.models import CacheScope, entry_id
from semantic_cache.service import SemanticCacheService
from semantic_cache.store import ChromaCacheStore
from tests.semantic_cache.smoke_local import QUERY, PARAPHRASE


def run_smoke(path):
    cache = SemanticCacheService(OllamaEmbeddingProvider(), ChromaCacheStore(path / 'cache'))
    assert type(agents.DDGS()) is ConcreteDDGS, 'Search tool does not construct the expected concrete DDGS class'
    report_path = Path(__file__).resolve().parents[2] / 'validation_logs' / 'phase6_research_regression.json'
    report_path.parent.mkdir(exist_ok=True)
    report = dict(ddgs_target='ddgs.ddgs.DDGS.text', crew_target='crewai.crew.Crew.kickoff',
                  cache_path=str(cache.store.path), initial_cache_count=cache.store.count(),
                  threshold=cache.similarity_threshold, requests=[])

    def checkpoint():
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    checkpoint()
    assert report['initial_cache_count'] == 0, 'Temporary cache must start empty'
    counts = dict(ddgs=0, crew=0, rag=0)
    originals = dict(ddgs=ConcreteDDGS.text, crew=agents.Crew.kickoff,
                     rag=agents.retrieve_document_context)
    lookups, writes = [], []
    original_lookup, original_save = cache.lookup, cache.save

    def observed_lookup(*args, **kwargs):
        result = original_lookup(*args, **kwargs)
        hit = result.hit
        lookups.append(dict(status='HIT' if hit else 'MISS', kind=hit.kind if hit else None,
                            similarity=hit.similarity if hit else None,
                            distance=1-hit.similarity if hit else None))
        return result

    def observed_save(*args, **kwargs):
        result = original_save(*args, **kwargs)
        writes.append(result)
        return result

    def counted(name):
        def call(*args, **kwargs):
            counts[name] += 1
            return originals[name](*args, **kwargs)
        return call

    try:
        with (path / 'crew-output.log').open('w', encoding='utf-8') as log, redirect_stdout(log), \
             patch.object(ConcreteDDGS, 'text', counted('ddgs')), \
             patch.object(agents.Crew, 'kickoff', counted('crew')), \
             patch.object(agents, 'retrieve_document_context', counted('rag')), \
             patch.object(cache, 'lookup', observed_lookup), patch.object(cache, 'save', observed_save):
            for query in (QUERY, PARAPHRASE):
                before = counts.copy()
                lookup_start, write_start = len(lookups), len(writes)
                started = perf_counter()
                answer = agents.run_research(query, use_rag=False, use_cache=True, cache_service=cache)
                seconds = perf_counter() - started
                flushed = crewai_event_bus.flush(timeout=30.0)
                request = dict(query=query, seconds=seconds, counts={k: counts[k]-before[k] for k in counts},
                               lookups=lookups[lookup_start:], writes=writes[write_start:],
                               cache_count=cache.store.count(), answer=answer)
                report['requests'].append(request)
                checkpoint()
                assert flushed, 'Crew events did not flush'
                assert answer.strip() and not answer.startswith('Error:'), 'Research returned an empty/error answer'
                assert len(request['lookups']) == 1, f'Expected one cache lookup: {request["lookups"]}'
                if len(report['requests']) == 1:
                    assert request['lookups'][0]['status'] == 'MISS', 'First request must miss'
                    assert request['counts']['ddgs'] > 0, f'DDGS did not execute: {request["counts"]}'
                    assert request['counts']['crew'] == 1, f'Expected one Crew execution: {request["counts"]}'
                    assert request['writes'] == [True] and request['cache_count'] == 1, 'First answer was not stored'
                    stored = cache.store.get_exact(entry_id(QUERY, CacheScope(use_rag=False)))
                    assert stored is not None and stored.answer == answer, 'Stored answer differs from first response'
                else:
                    lookup = request['lookups'][0]
                    assert lookup['status'] == 'HIT' and lookup['kind'] == 'semantic', 'Second request must be a semantic hit'
                    assert lookup['similarity'] > cache.similarity_threshold, 'Hit did not exceed configured threshold'
                    assert all(value == 0 for value in request['counts'].values()), f'Cache hit ran downstream work: {request["counts"]}'
                    assert request['writes'] == [] and request['cache_count'] == 1, 'Cache hit unexpectedly wrote an entry'
                    assert answer == stored.answer == report['requests'][0]['answer'], 'Cached answer did not match exactly'
        report['answer_exactly_matched'] = True
        report['passed'] = True
        checkpoint()
    except Exception as error:
        report['passed'] = False
        report['failure'] = str(error)
        checkpoint()
        raise
    print(json.dumps(dict(report_path=str(report_path), ddgs_target=report['ddgs_target'],
                         initial_cache_count=report['initial_cache_count'], threshold=report['threshold'],
                         requests=[{k:v for k,v in request.items() if k != 'answer'} for request in report['requests']],
                         answer_exactly_matched=True, passed=True), indent=2))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        run_smoke(Path(sys.argv[2]))
    else:
        # Preserve the cache and Crew log even when a worker assertion fails.
        directory = tempfile.mkdtemp(prefix='research_cache_e2e_')
        print(f'Regression temporary directory (retained): {directory}', flush=True)
        subprocess.run([sys.executable, '-m', 'tests.semantic_cache.smoke_research',
                        '--worker', directory], check=True, timeout=1200)
