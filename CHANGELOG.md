# Release notes

## v1.0.0 candidate — Multi-Agent Deep Researcher

This documents the completed local project for review; it does not create a Git tag or remote release.

- Three sequential CrewAI research agents with real DDGS search, local Ollama inference, and an MCP stdio tool.
- Local PDF/TXT/Markdown RAG with Qwen3 embeddings, persistent Chroma, bounded context, and document citations.
- Scope-aware exact/semantic caching with TTL and guarded reuse; deterministic FAST/QUALITY model selection.
- Request-scoped execution traces, SQLite operational metrics, and a Streamlit Research Execution Console with Analytics.
- Versioned evaluation datasets, isolated/checkpointed measurements, and an authoritative [Phase 10 baseline](docs/benchmarks/phase10_baseline.md).
- Final portfolio landing page, Mermaid architecture, setup/evaluation instructions, demo checklist, and defensible resume/interview material.

Phase 11 changes documentation and runtime-artifact exclusions only. Models, prompts, routing/cache/RAG settings, orchestration, metrics schema, UI behavior, dependency pins, and committed benchmark evidence remain unchanged. No new research generation or DDGS call is part of this release-readiness work.

Measured results retain their sample sizes and limitations; see [README](README.md) and [the portfolio](docs/portfolio.md). Review before choosing an optional `v1.0.0` tag or push. No repository license has been added.
