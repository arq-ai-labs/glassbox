# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha] - 2026-03-30

### Added

- **`wrap()`** — One-line integration for OpenAI and Anthropic clients. Captures a ContextPack for every LLM call automatically.
- **`observe()`** — One-line integration for LangGraph agents. Captures every LLM call across multi-step graph execution.
- **`serve()`** — Built-in viewer UI at `localhost:4100`. Three-column layout: timeline, step detail, metadata sidebar.
- **ContextPack format v0.1.0** — Structured JSON capturing sections, token budget, model info, output, and metrics per LLM call.
- **8 section types** — system_prompt, user_message, assistant_message, tool_result, retrieval, memory, instruction, custom.
- **Token budget with rejection ledger** — Tracks what was included, what was excluded, and why (7 rejection reasons).
- **Context assembler** — Budget-aware context builder with automatic rejection tracking.
- **Multi-agent delegation links** — Captures parent_run_id, parent_step_id, delegation_scope, and inherited_sections across agent handoffs.
- **3 conformance levels** — L1 Capture, L2 Budget, L3 Multi-Agent.
- **4 redaction policies** — NONE, HASH, TRUNCATE, DROP_CONTENT for privacy-sensitive deployments.
- **Cost estimation** — 40+ model pricing table covering OpenAI, Anthropic, Google, Meta, and Mistral models.
- **Context drift detection** — Step-over-step section diffing to track how context evolves across an agent run.
- **ContextPack JSON Schema** — Machine-validatable schema published alongside the spec.
- **155 passing tests** — Covering format models, conformance, diffing, assembly, redaction, pricing, storage, emitter, wrap(), and observe().
- **Spec + fixtures** — CONTEXTPACK_SPEC.md with 9 conformance fixtures (5 valid, 4 invalid).

[0.1.0-alpha]: https://github.com/arq-ai-labs/glassbox/releases/tag/v0.1.0-alpha
