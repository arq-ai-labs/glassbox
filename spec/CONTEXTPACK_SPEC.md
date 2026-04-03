# ContextPack Format Specification

**Version:** 0.1.0
**Status:** Draft
**Author:** Habib Mehmoodi
**Date:** 2026-03-30

## 1. Introduction

ContextPack is a structured, versioned format for recording exactly what context an LLM received during a single inference call — what was included, what was excluded and why, which model was used, and what it produced.

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://datatracker.ietf.org/doc/html/rfc2119).

## 2. Design Goals

1. **Observable:** Every LLM call produces a complete, self-contained record.
2. **Auditable:** Rejection ledger captures what was excluded and why.
3. **Interoperable:** JSON-based, language-agnostic, validatable via JSON Schema.
4. **Composable:** Works alongside OpenTelemetry, not instead of it.
5. **Privacy-aware:** Content can be redacted while preserving structure.

## 3. Format Overview

A ContextPack is a JSON object with the following top-level structure:

```
ContextPack
├── Envelope (format_version, run/step identity, timestamps)
├── Sections[] (discriminated union of context types)
├── TokenBudget (allocation ledger + rejection ledger)
├── ModelInfo (provider, model, parameters)
├── OutputRecord (what the model produced)
├── Metrics (latency, tokens, cost)
├── MultiAgentLink? (delegation metadata)
└── Extensions? (vendor-specific data)
```

## 4. Required Fields

An implementation MUST include the following fields in every ContextPack:

| Field | Type | Description |
|-------|------|-------------|
| `format_version` | string | Semantic version of the ContextPack format. MUST be "0.1.0" for this spec version. |
| `run_id` | string | Unique identifier for the run (session/conversation). |
| `step_id` | string | Unique identifier for this specific LLM call within the run. |
| `step_index` | integer (≥ 0) | Zero-based ordinal position within the run. |
| `started_at` | string | ISO 8601 timestamp with millisecond precision and Z suffix. |
| `completed_at` | string | ISO 8601 timestamp with millisecond precision and Z suffix. |
| `sections` | array | Ordered list of context sections (see §5). |
| `token_budget` | object | Token allocation and rejection ledger (see §6). |
| `model` | object | Model identity and parameters (see §7). |
| `output` | object | What the model produced (see §8). |
| `metrics` | object | Performance data (see §9). |

## 5. Sections

Sections are the atomic units of context. Each section represents one semantically distinct piece of information sent to the model.

### 5.1 Section Base Fields

Every section MUST include:

| Field | Type | Description |
|-------|------|-------------|
| `section_id` | string | Stable identifier within this ContextPack. |
| `type` | string | Discriminator. MUST be one of the registered types. |
| `token_count` | integer (≥ 0) | Token count for this section's content. |
| `content` | string | The text content. MAY be redacted (see §12). |

Every section MAY include:

| Field | Type | Description |
|-------|------|-------------|
| `source` | string | Human-readable origin label (e.g., "system", "tool:search", "rag:pinecone"). |
| `metadata` | object | Arbitrary key-value pairs for vendor-specific data. |

### 5.2 Registered Section Types

| Type | Description | Additional Fields |
|------|-------------|-------------------|
| `system_prompt` | System instructions to the model. | — |
| `user_message` | User/human input. | — |
| `assistant_message` | Prior assistant output (in multi-turn). | — |
| `tool_result` | Output from a tool call. | `tool_name` (REQUIRED), `tool_call_id` |
| `retrieval` | RAG/search result. | `query`, `source_collection`, `score` (0-1), `document_id` |
| `memory` | Conversation/long-term/working memory. | `memory_type` (conversation\|long_term\|working\|episodic) |
| `instruction` | Runtime instructions/guardrails. | — |
| `custom` | Extension point. | `custom_type` (REQUIRED) |

An implementation MUST reject sections with unrecognized `type` values (strict validation). Future spec versions MAY add new types.

### 5.3 Extension Mechanism

The `custom` section type with a `custom_type` discriminator allows vendors to define application-specific sections. Custom types SHOULD use a namespaced format: `vendor:type_name` (e.g., `myapp:guardrail_result`).

## 6. Token Budget

The token budget records how the context window was allocated and what was excluded.

### 6.1 Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `total_budget` | integer (≥ 0) | Maximum tokens available for context. |
| `total_used` | integer (≥ 0) | Tokens consumed by included sections. |
| `by_section` | array | Per-section allocation: `{section_id, section_type, token_count}`. |

### 6.2 Rejection Ledger

| Field | Type | Description |
|-------|------|-------------|
| `rejected` | array | Candidates that were considered but excluded. |

Each rejected candidate contains:

| Field | Type | Description |
|-------|------|-------------|
| `section_type` | string | What kind of section this would have been. |
| `token_count` | integer (≥ 0) | Tokens this candidate would have consumed. |
| `reason` | string | MUST be one of the registered rejection reasons. |
| `source` | string | OPTIONAL. Origin label. |
| `reason_detail` | string | OPTIONAL. Human-readable explanation. |

### 6.3 Rejection Reasons

| Reason | Meaning |
|--------|---------|
| `token_limit_exceeded` | Section would exceed the remaining token budget. |
| `relevance_below_threshold` | Retrieval/relevance score below cutoff. |
| `staleness` | Data is older than the freshness threshold. |
| `redundant` | Equivalent content already included. |
| `policy_excluded` | Excluded by a policy or guardrail rule. |
| `priority_displaced` | Lower-priority content displaced by higher-priority. |
| `custom` | Application-specific reason (explain in `reason_detail`). |

An implementation MUST reject rejection reasons not in this list.

## 7. Model Info

| Field | Type | Description |
|-------|------|-------------|
| `provider` | string | REQUIRED. Provider name: "openai", "anthropic", "azure", "ollama", etc. |
| `model` | string | REQUIRED. Full model identifier (e.g., "claude-sonnet-4-20250514"). |
| `temperature` | number | OPTIONAL. |
| `max_tokens` | integer | OPTIONAL. |
| `top_p` | number | OPTIONAL. |
| `stop_sequences` | array of strings | OPTIONAL. |
| `additional_params` | object | OPTIONAL. Vendor-specific parameters. |

## 8. Output Record

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | REQUIRED. One of: "text", "tool_calls", "mixed", "error". |
| `text` | string | OPTIONAL. Generated text content. |
| `tool_calls` | array | OPTIONAL. Each: `{tool_call_id, tool_name, arguments}`. |
| `stop_reason` | string | OPTIONAL. Why generation stopped. |
| `error` | string | OPTIONAL. Error message if type is "error". |

## 9. Metrics

| Field | Type | Description |
|-------|------|-------------|
| `latency_ms` | number (≥ 0) | REQUIRED. Wall-clock time in milliseconds. |
| `input_tokens` | integer (≥ 0) | REQUIRED. Tokens consumed by input. |
| `output_tokens` | integer (≥ 0) | REQUIRED. Tokens generated in output. |
| `cache_read_tokens` | integer (≥ 0) | OPTIONAL. Tokens served from cache. |
| `cache_creation_tokens` | integer (≥ 0) | OPTIONAL. Tokens written to cache. |
| `cost_estimate_usd` | number (≥ 0) | OPTIONAL. Estimated cost in USD. |

## 10. Multi-Agent Link

OPTIONAL. Present when this step is part of a multi-agent delegation.

| Field | Type | Description |
|-------|------|-------------|
| `parent_run_id` | string | Run ID of the delegating agent. |
| `parent_step_id` | string | Step ID that initiated delegation. |
| `delegation_scope` | string | Description of the delegated task. |
| `inherited_sections` | array of strings | Section IDs inherited from parent. |

## 11. Versioning

### 11.1 Compatibility Rules

- **Patch versions** (0.1.x): Bug fixes to this document. No schema changes.
- **Minor versions** (0.x.0): Additive changes only. New optional fields, new section types. Existing ContextPacks MUST remain valid.
- **Major versions** (x.0.0): Breaking changes. Existing ContextPacks MAY become invalid.

### 11.2 Forward Compatibility

A reader encountering a ContextPack with a higher minor version than it supports SHOULD:
1. Parse all known fields.
2. Preserve unknown fields in `extensions` or ignore them.
3. NOT reject the ContextPack solely due to unknown optional fields.

A reader encountering a higher major version SHOULD reject the ContextPack with a clear error.

## 12. Privacy and Redaction

Implementations SHOULD provide a redaction mechanism that processes sections before storage. Recommended strategies:

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `none` | Store full content. | Development, non-sensitive data. |
| `hash` | Replace content with `sha256:<hex>`. | Audit trails without raw content. |
| `truncate` | Keep first N characters. | Debugging with minimal exposure. |
| `drop_content` | Empty string, preserve metadata. | Compliance-only logging. |

Redaction MUST preserve: `section_id`, `type`, `source`, `token_count`, and `metadata`. Redaction MUST NOT modify fields outside the `sections` array.

## 13. File Storage Layout

The RECOMMENDED file layout for on-disk storage is:

```
.glassbox/
├── index.json                          # Run index
└── runs/
    └── <run_id>/
        ├── run.json                    # RunMetadata
        └── steps/
            ├── 0000_<step_id>.json     # ContextPack
            ├── 0001_<step_id>.json
            └── ...
```

Step files MUST be named `{step_index:04d}_{step_id}.json` for lexicographic ordering.

## 14. Conformance

### 14.1 Conformance Levels

| Level | Requirements |
|-------|-------------|
| **Level 1: Capture** | Produce valid ContextPacks with sections, token_budget, model, output, metrics. |
| **Level 2: Budget** | Level 1 + populate rejection ledger when context assembly decisions are made. |
| **Level 3: Multi-Agent** | Level 2 + populate multi_agent links in delegated agent architectures. |

### 14.2 Conformance Testing

A conformance test suite is provided as JSON fixtures in `spec/fixtures/`. An implementation is conformant at Level 1 if:

1. All `valid_*.json` fixtures parse without error.
2. All `invalid_*.json` fixtures produce a validation error.
3. Valid fixtures survive a serialize → deserialize round-trip with no data loss.

## 15. Relationship to OpenTelemetry

ContextPack is complementary to OpenTelemetry, not a replacement. A ContextPack captures *what context an LLM saw* — OTel captures *how the system performed*.

Implementations MAY emit ContextPacks as:
- OTel span events (attach to the LLM call span)
- OTel span attributes (for key metrics like token counts)
- OTel log records (for the full ContextPack JSON)

The `run_id` and `step_id` fields SHOULD be correlated with OTel trace and span IDs when both systems are active.

## 16. JSON Schema

The authoritative JSON Schema for ContextPack 0.1.0 is published at:
`spec/context-pack.schema.json`

This schema is auto-generated from the reference implementation but is the normative definition. In case of conflict between this specification document and the JSON Schema, this document takes precedence.

---

## Appendix A: Full Example

See `spec/fixtures/valid_full.json` for a complete ContextPack demonstrating all features: 6 section types, 3 rejection reasons, multi-agent links, extensions, and cost estimation.

## Appendix B: Changelog

### 0.1.0 (2026-03-30)
- Initial specification.
- 8 section types, 7 rejection reasons, 3 conformance levels.
- JSON Schema published.
- Conformance test fixtures (5 valid, 4 invalid).
