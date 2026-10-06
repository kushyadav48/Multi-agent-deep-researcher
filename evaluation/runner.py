"""Sequential, checkpointed benchmark orchestration with durable call limits."""

from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import mkdtemp
from time import perf_counter, process_time
from unittest.mock import patch

from evaluation import BENCHMARK_VERSION
from evaluation import cache_eval, rag_eval, router_eval, synthesis_eval
from evaluation.datasets import ROOT, evidence, load
from evaluation.metrics import cache_metrics, cache_speedup, latency_summary, rag_metrics, routing_metrics
from evaluation.models import BenchmarkCaseResult, BenchmarkRun, BenchmarkSuiteResult
from rag.context import DEFAULT_MAX_DISTANCE, DEFAULT_TOP_K
from rag.embeddings import OllamaEmbeddingProvider
from routing.models import FAST_SYNTHESIS_MODEL, QUALITY_SYNTHESIS_MODEL, ROUTER_POLICY_VERSION, SEARCH_MODEL
from routing.router import QUALITY_THRESHOLD, route_query
from semantic_cache.service import DEFAULT_SIMILARITY_THRESHOLD, DEFAULT_TTL_SECONDS, SemanticCacheService
from semantic_cache.store import ChromaCacheStore


PROJECT = ROOT.parent
MAX_RESEARCH_INVOCATIONS = 8
MAX_CREW_EXECUTIONS = 7
LIVE_QUERY = "What is the Model Context Protocol (MCP)? Give a concise explanation."
CACHE_QUERY = "What is the Model Context Protocol?"
CACHE_PARAPHRASE = "Explain the Model Context Protocol."
SUITES = {
    "offline": ("router", "cache_pairs", "rag_retrieval"),
    "controlled": ("synthesis", "cache_e2e", "rag_integration"),
    "live-web": ("live_web",),
}


class BudgetExceeded(RuntimeError):
    pass


class LiveWebBudgetExceeded(RuntimeError):
    pass


def source_hash():
    paths = list(ROOT.glob("*.py")) + list((ROOT / "datasets").glob("*.json"))
    paths += [path for path in (ROOT / "fixtures").rglob("*") if path.suffix in (".json", ".md")]
    paths += [PROJECT / "agents.py"]
    for package in ("routing", "semantic_cache", "rag", "observability"):
        paths.extend((PROJECT / package).glob("*.py"))
    digest = sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(PROJECT).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def configuration():
    from semantic_cache.models import CACHE_VERSION, EMBEDDING_MODEL
    return dict(searcher=SEARCH_MODEL, fast_synthesis=FAST_SYNTHESIS_MODEL,
                quality_synthesis=QUALITY_SYNTHESIS_MODEL, embedding_model=EMBEDDING_MODEL,
                router_policy=ROUTER_POLICY_VERSION, quality_threshold=QUALITY_THRESHOLD,
                cache_similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
                cache_ttl_seconds=DEFAULT_TTL_SECONDS, cache_version=CACHE_VERSION,
                rag_top_k=DEFAULT_TOP_K, rag_max_distance=DEFAULT_MAX_DISTANCE,
                max_research_invocations=MAX_RESEARCH_INVOCATIONS, max_crew_executions=MAX_CREW_EXECUTIONS)


@contextmanager
def output_lock(output):
    """OS-released advisory lock: a crashed process cannot strand a lock."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Runner:
    def __init__(self, output, *, resume=False, research=None, provider_factory=None, progress=print):
        self.output = Path(output).resolve()
        if self.output == PROJECT or self.output == ROOT or PROJECT / "data" in (self.output, *self.output.parents):
            raise ValueError("Benchmark output must be a dedicated directory outside production data")
        self.output.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.output / "checkpoint.json"
        self.research = research
        self.provider_factory = provider_factory or OllamaEmbeddingProvider
        self.progress = progress
        self.state = {}
        self._rag = None
        self._cache = None
        if self.checkpoint.exists():
            if not resume:
                raise ValueError("Existing checkpoint: use --resume or a new output directory")
            saved = json.loads(self.checkpoint.read_text(encoding="utf-8"))
            self.run_result = BenchmarkRun.from_dict(saved["run"])
            self.state = saved["state"]
            if (self.run_result.benchmark_version != BENCHMARK_VERSION or
                self.run_result.source_hash != source_hash() or self.run_result.configuration != configuration()):
                raise ValueError("Checkpoint methodology/configuration does not match this harness")
            for suite in self.run_result.suites:
                for case in suite.cases:
                    if case.status == "IN_PROGRESS":
                        case.status = "INTERRUPTED"
                        case.errors = ["InterruptedExecution"]
                        case.warnings = ["Reserved call is not automatically retried; actual completion is unknown."]
        else:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
            self.run_result = BenchmarkRun(datetime.now(timezone.utc).isoformat(), commit,
                dict(python=sys.version.split()[0], platform=platform.platform(),
                     cpu_measurement="Python process CPU time only; excludes Ollama server and GPU"),
                configuration(), source_hash())
        self.save()

    def save(self):
        self.summarize()
        temporary = self.checkpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(run=self.run_result.to_dict(), state=self.state),
                                         indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.checkpoint)

    def suite(self, name):
        existing = next((suite for suite in self.run_result.suites if suite.name == name), None)
        if existing is None:
            category = next(key for key, values in SUITES.items() if name in values)
            existing = BenchmarkSuiteResult(name, category.upper())
            self.run_result.suites.append(existing)
        return existing

    def find(self, identity):
        return next((case for suite in self.run_result.suites for case in suite.cases if case.case_id == identity), None)

    def scratch(self, identity):
        if identity not in self.state:
            parent = self.output / "state"
            parent.mkdir(exist_ok=True)
            self.state[identity] = mkdtemp(prefix=f"{identity}_", dir=parent)
            self.save()
        path = Path(self.state[identity]).resolve()
        if self.output / "state" not in path.parents:
            raise ValueError("Checkpoint store path escaped benchmark state directory")
        return path

    def rag(self):
        if self._rag is None:
            self._rag = rag_eval.prepare(self.provider_factory(), self.scratch("rag"))
        return self._rag

    def cache(self):
        if self._cache is None:
            self._cache = SemanticCacheService(self.provider_factory(), ChromaCacheStore(self.scratch("cache_e2e")))
        return self._cache

    def case(self, identity, suite, action, *, expensive=False):
        if self.find(identity) is not None:
            self.progress(f"Resume: retained {identity}", flush=True)
            return
        if expensive:
            if self.run_result.research_reservations >= MAX_RESEARCH_INVOCATIONS:
                raise BudgetExceeded("Research invocation cap reached")
            if identity != "cache-hit" and self.run_result.crew_reservations >= MAX_CREW_EXECUTIONS:
                raise BudgetExceeded("Crew execution cap reached")
        placeholder = BenchmarkCaseResult(identity, suite, "IN_PROGRESS")
        self.suite(suite).cases.append(placeholder)
        if expensive:
            # Reserve before calling: a process failure cannot cause a repeat.
            self.run_result.research_reservations += 1
            if identity != "cache-hit":
                self.run_result.crew_reservations += 1
        self.save()
        self.progress(f"Starting {identity}", flush=True)
        started = perf_counter()
        try:
            result = action()
            if result.case_id != identity or result.suite != suite:
                raise ValueError("Evaluator identity mismatch")
        except Exception as error:
            result = BenchmarkCaseResult(identity, suite, "ERROR",
                                         duration_ms=(perf_counter() - started) * 1000,
                                         errors=[type(error).__name__])
        self.suite(suite).cases[-1] = result
        self.run_result.crew_executions += result.measured.get("crew_executions", 0)
        self.save()
        self.progress(f"Completed {identity}: {result.status}", flush=True)

    def execute(self, identity, suite, query, *, route="auto", fixed=None, use_rag=False,
                use_cache=False, concepts=(), preflight_hit=False):
        import agents
        from crewai.events import crewai_event_bus
        from ddgs.ddgs import DDGS
        from observability.store import MetricsStore
        kwargs = dict(use_rag=use_rag, use_cache=use_cache, model_route=route,
                      metrics_store=MetricsStore(self.scratch("metrics") / "metrics.db"))
        if use_rag:
            kwargs["rag_service"] = self.rag()
        if use_cache:
            kwargs["cache_service"] = self.cache()
        if preflight_hit:
            scope = agents.research_cache_scope(None, use_rag=False, top_k=DEFAULT_TOP_K,
                max_distance=DEFAULT_MAX_DISTANCE, routing_decision=route_query(query, route))
            lookup = self.cache().lookup(query, scope)
            if lookup.hit is None or lookup.hit.kind != "semantic":
                # Refuse an accidental second generation. Record the miss.
                return BenchmarkCaseResult(identity, suite, "FAIL", measured=dict(
                    preflight_semantic_hit=False, crew_executions=0, research_not_invoked=True),
                    warnings=["Semantic hit preflight failed; generation was not retried."])
        originals = DDGS.text
        network = dict(real_ddgs_calls=0, blocked_ddgs_calls=0)

        def one_real_call(*args, **call_kwargs):
            if network["real_ddgs_calls"] >= 1:
                network["blocked_ddgs_calls"] += 1
                raise LiveWebBudgetExceeded()
            network["real_ddgs_calls"] += 1
            return originals(*args, **call_kwargs)

        boundary = (patch.object(DDGS, "text", side_effect=lambda *args, **kw: deepcopy(fixed))
                    if fixed is not None else patch.object(DDGS, "text", new=one_real_call)
                    if suite == "live_web" else nullcontext())
        current = self.find(identity)
        if current is None or current.status != "IN_PROGRESS":
            raise ValueError("Research requires an active checkpoint reservation")
        if self.run_result.research_invocations >= MAX_RESEARCH_INVOCATIONS:
            raise BudgetExceeded("Research invocation cap reached")
        self.run_result.research_invocations += 1
        self.save()
        cpu_started = process_time()
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink), boundary:
            result = (self.research or agents.run_research_detailed)(query, **kwargs)
            # Flush verbose rendering before restoring the console; do not retain logs.
            try:
                drained = crewai_event_bus.flush(timeout=30)
            except Exception:
                drained = False
        measured = synthesis_eval.evaluate(identity, suite, result, requested_route=route,
            concepts=concepts, urls=[row["href"] for row in fixed] if fixed else ())
        measured.measured["python_process_cpu_ms"] = (process_time() - cpu_started) * 1000
        if fixed is not None and result.cache.status not in ("EXACT_HIT", "SEMANTIC_HIT"):
            expected_results = [(row["title"], row["href"], row["body"]) for row in fixed]
            verified = bool(result.web.calls) and all(
                [(row.title, row.url, row.snippet) for row in call.results] == expected_results
                for call in result.web.calls)
            measured.measured["fixed_evidence_sha256"] = sha256(json.dumps(fixed, sort_keys=True).encode()).hexdigest()
            measured.measured["fixed_evidence_verified"] = verified
            if not verified and measured.status == "PASS":
                measured.status = "FAIL"
        if suite == "live_web":
            measured.measured.update(network)
        if not drained:
            measured.warnings.append("Crew event rendering did not finish within the drain timeout.")
        if suite == "rag_integration":
            checks = dict(rag_used=result.rag.used,
                expected_source_retrieved=any(item.source == "project_meridian.md" for item in result.rag.evidence),
                fact_in_evidence=any("Cedar-47" in item.text for item in result.rag.evidence),
                fact_in_answer="cedar-47" in result.final_answer.casefold(),
                local_citation="[Document: project_meridian.md]" in result.final_answer)
            measured.measured["integration_checks"] = checks
            if not all(checks.values()) and measured.status == "PASS":
                measured.status = "FAIL"
        if suite == "cache_e2e":
            measured.measured["answer_matches_cold"] = (identity == "cache-hit" and
                self.find("cache-cold").measured.get("answer_sha256") == measured.measured["answer_sha256"])
        return measured

    def run(self, selection):
        if selection not in (*SUITES, "all"):
            raise ValueError("Unknown suite")
        selected = tuple(SUITES) if selection == "all" else (selection,)
        if "offline" in selected:
            for item in load("routing"):
                self.case(item["id"], "router", lambda item=item: router_eval.evaluate(item))
            for item in load("semantic_cache"):
                self.case(item["id"], "cache_pairs", lambda item=item: cache_eval.evaluate(
                    item, self.provider_factory(), self.scratch(item["id"])))
            for item in load("rag"):
                self.case(item["id"], "rag_retrieval", lambda item=item: rag_eval.evaluate(item, self.rag()))
        if "controlled" in selected:
            for item in load("synthesis"):
                for route in ("fast", "quality"):
                    identity = f"{item['id']}-{route}"
                    self.case(identity, "synthesis", lambda item=item, route=route, identity=identity: self.execute(
                        identity, "synthesis", item["query"], route=route, fixed=evidence(item["evidence"]),
                        concepts=item["concept_groups"]), expensive=True)
            self.case("cache-cold", "cache_e2e", lambda: self.execute(
                "cache-cold", "cache_e2e", CACHE_QUERY, fixed=evidence("mcp"), use_cache=True), expensive=True)
            self.case("cache-hit", "cache_e2e", lambda: self.execute(
                "cache-hit", "cache_e2e", CACHE_PARAPHRASE, fixed=evidence("mcp"), use_cache=True,
                preflight_hit=True), expensive=True)
            self.case("rag-integration", "rag_integration", lambda: self.execute(
                "rag-integration", "rag_integration", "What is Project Meridian's internal protocol codename?",
                fixed=evidence("rag_context"), use_rag=True), expensive=True)
        if "live-web" in selected:
            self.case("live-web-mcp", "live_web", lambda: self.execute(
                "live-web-mcp", "live_web", LIVE_QUERY), expensive=True)
        self.save()
        return self.run_result

    def summarize(self):
        for suite in self.run_result.suites:
            measured = [dict(case_id=case.case_id, **case.measured) for case in suite.cases if case.status in ("PASS", "FAIL")]
            if suite.name == "router":
                suite.metrics = routing_metrics(measured)
            elif suite.name == "cache_pairs":
                suite.metrics = cache_metrics(measured)
            elif suite.name == "rag_retrieval":
                suite.metrics = rag_metrics(measured, DEFAULT_TOP_K)
            elif suite.name == "synthesis":
                suite.metrics = {route: latency_summary([case.timings["total_ms"] for case in suite.cases
                    if case.status in ("PASS", "FAIL") and case.route == route and case.timings.get("total_ms") is not None])
                    for route in ("FAST", "QUALITY")}
            elif suite.name == "cache_e2e":
                cold, hit = self.find("cache-cold"), self.find("cache-hit")
                suite.metrics = cache_speedup(cold.measured, hit.measured) if cold and hit and all(
                    "total_ms" in case.measured for case in (cold, hit)) else {}
            suite.metrics["attempted_cases"] = len(suite.cases)
            suite.metrics["execution_errors"] = [case.case_id for case in suite.cases
                if case.status not in ("PASS", "FAIL", "IN_PROGRESS")]
