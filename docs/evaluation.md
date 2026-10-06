# Evaluation and reproduction

The authoritative measurements are [the Phase 10 Markdown report](benchmarks/phase10_baseline.md) and [baseline JSON](../evaluation/results/phase10_baseline.json), benchmark version `phase10-v1`. These committed artifacts remain unchanged during Phase 11. Reading them requires no Ollama, DDGS, or benchmark execution.

## What each suite measures

| Suite | Work | Requirements |
| --- | --- | --- |
| `offline` | 36 curated routing queries, 20 cache pairs, 6 synthetic documents / 10 retrieval queries | Ollama embedding model; no research generation or DDGS network |
| `controlled` | Two prompts forced through FAST and QUALITY, a cold/cache-hit pair, one RAG integration case | All three local models; fixed DDGS evidence replaces the concrete network boundary |
| `live-web` | One Auto research request with at most one real DDGS call | All models available, internet access; embeddings used only if the selected case needs them |

Controlled evidence makes each prompt's source material identical across routes. Its deterministic concept-group rubric measures content presence rather than factual correctness, citation entailment, or human-rated quality. Offline cache false negatives are measured classification outcomes, not broken regression tests.

The recorded router outcome was 100% policy-conformance on 36 curated queries. The 20 cache pairs yielded TP=4, TN=10, FP=0, FN=6: 100% precision, 40% recall, 70% accuracy. On the six-document, ten-query synthetic corpus, RAG achieved 100% Hit@1/Hit@4 and MRR 1.0, with mean retrieval time 50.32 ms including query embedding/store access and excluding ingestion.

The one measured cache paraphrase pair took 32.91 s cold versus 32.98 ms on semantic reuse, a 997.95× speedup and 99.90% latency reduction. Cold denotes an empty cache, not necessarily an unloaded generation model. RAG was disabled in both requests; the hit avoided one Crew and four fixed-evidence web tool calls. These figures are not a universal performance or cost-saving estimate.

FAST and QUALITY each had only two controlled prompts: mean/median latency 36.47 s and 28.26 s respectively; mean reported tokens 6,477.5 and 4,449.5. Both covered all rubric groups. QUALITY was faster/lower-token in this run; the sample cannot establish a general speed or quality ranking. There was one sample per prompt/route, FAST before QUALITY, no fixed generation seed or generation warm-up, and varying model warmth/load.

## Safe regression and interface checks

With the environment described in [README](../README.md), install pytest if absent, then run:

```powershell
python -m pytest -q
python -m pip check
python -m evaluation.cli --help
python -m tests.smoke_interfaces
```

Pytest uses fakes/mocks and isolated stores. Interface smoke checks HTTP root/health plus raw MCP initialize/tools-list/disconnect; it never invokes research. The other opt-in `smoke_research` modules perform real research and are not needed to review the portfolio or validate documentation.

## Optional new measurements

Use a new dedicated output directory for a deliberate new run. The following PowerShell commands use the virtual environment explicitly and do not overwrite the committed baseline:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m evaluation.cli --suite offline --output validation_logs/phase10-new
.\.venv\Scripts\python.exe -X utf8 -m evaluation.cli --suite controlled --output validation_logs/phase10-new --resume
.\.venv\Scripts\python.exe -X utf8 -m evaluation.cli --suite live-web --output validation_logs/phase10-new --resume
```

Controlled/live-web calls can take substantial local inference time; they are optional and unnecessary for a demo. `--suite all` selects all categories. A full run allows at most eight research reservations and seven Crew reservations; controlled uses seven requests and normally six Crew executions, while live-web permits one request and one real DDGS boundary call. These caps apply to requests/Crews, not every internal model call.

Isolated Chroma, cache, SQLite, and atomic checkpoints live beneath the ignored output directory. `--resume` retains successful, failed, and interrupted cases without automatically repeating a generation. A source/configuration hash rejects incompatible checkpoints. An OS advisory lock prevents concurrent use of an output directory. Cache-hit preflight refuses an accidental second generation if semantic reuse fails.

## Report-only regeneration

With an existing compatible checkpoint, regenerate reports without embedding, generation, or DDGS calls:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m evaluation.cli --suite all --output validation_logs/phase10-new --resume --report-only --report-json validation_logs/phase10-new/results.json --report-markdown validation_logs/phase10-new/report.md
```

The committed JSON/Markdown are sufficient for reviewing Phase 10; they are report outputs, not resumable checkpoints. Reports preserve operational measurements/model identities, hashes, retrieval ranks, and rubric outcomes while excluding generated answers, prompts, agent outputs, vectors, credentials, and verbose logs.

Timings come from the production trace or offline `perf_counter`. Research totals exclude metrics persistence, harness setup, preflight, and event draining. Web time overlaps Crew time. Python process CPU excludes the Ollama service/GPU. The single live-web result (44.61 s, 5 results, 5,281 tokens) is not average latency or an availability measure. See [the report's limitations](benchmarks/phase10_baseline.md#10-limitations) before quoting any number.
