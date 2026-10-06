"""Explicit CLI: normal pytest never executes a real benchmark."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from evaluation.reporting import write_reports
from evaluation.runner import Runner, output_lock


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen Phase 10 evaluation. Offline uses real local embeddings, no generation/web.")
    parser.add_argument("--suite", choices=("offline", "controlled", "live-web", "all"), default="offline",
                        help="Controlled: 7 research calls, up to 6 Crews with fixed DDGS evidence. Live-web: 1 call, at most 1 real DDGS call.")
    parser.add_argument("--output", type=Path, default=Path("validation_logs/phase10"), help="Dedicated scratch/checkpoint directory")
    parser.add_argument("--resume", action="store_true", help="Retain all completed/failed cases; never retry generation")
    parser.add_argument("--report-only", action="store_true", help="Regenerate reports from checkpoint; requires --resume")
    parser.add_argument("--report-json", type=Path, help="JSON destination; default OUTPUT/results.json")
    parser.add_argument("--report-markdown", type=Path, help="Markdown destination; default OUTPUT/report.md")
    args = parser.parse_args(argv)
    if args.report_only and (not args.resume or not (args.output / "checkpoint.json").is_file()):
        parser.error("--report-only requires --resume and an existing checkpoint")
    with output_lock(args.output.resolve()):
        runner = Runner(args.output, resume=args.resume)
        if not args.report_only:
            if "local_models" not in runner.run_result.environment:
                try:
                    with urlopen("http://localhost:11434/api/tags", timeout=10) as response:
                        tags = json.load(response)
                    names = {value.removeprefix("ollama/") for key, value in runner.run_result.configuration.items()
                             if key in ("searcher", "fast_synthesis", "quality_synthesis", "embedding_model")}
                    runner.run_result.environment["local_models"] = [{
                        "name": model["name"], "digest": model.get("digest"),
                        "size_bytes": model.get("size"), "quantization": model.get("details", {}).get("quantization_level"),
                    } for model in tags["models"] if model["name"] in names]
                except Exception as error:
                    runner.run_result.environment["model_metadata_error_type"] = type(error).__name__
            runner.run(args.suite)
        write_reports(runner.run_result, args.report_json or runner.output / "results.json",
                      args.report_markdown or runner.output / "report.md")
        print(json.dumps(dict(benchmark_version=runner.run_result.benchmark_version,
                             research_invocations=runner.run_result.research_invocations,
                             research_reservations=runner.run_result.research_reservations,
                             crew_executions=runner.run_result.crew_executions,
                             suites={suite.name: suite.metrics for suite in runner.run_result.suites}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
