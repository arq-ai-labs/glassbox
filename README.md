# Glassbox

**AI context observability. See what your LLM saw.**

Glassbox captures a **ContextPack** for every LLM call your agent makes — what context went in, what got excluded and why, which model was used, what came out, and what it cost. One line of code. Zero config.

```bash
pip install glassbox-ctx[openai,server]
```

```python
from glassbox import wrap, serve
from openai import OpenAI

client = wrap(OpenAI())

# Use the client exactly as before — every call is now captured
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is context engineering?"},
    ]
)

serve()  # Open http://localhost:4100
```

<!-- TODO: Add viewer screenshot here -->
<!-- ![Glassbox Viewer](docs/assets/viewer-screenshot.png) -->

---

## Why

You're building AI agents. They call LLMs dozens of times per run. Each call gets a different context window — different system prompts, retrieved documents, tool results, conversation history. When something goes wrong, you're staring at logs trying to reconstruct what the model actually saw.

Glassbox records the full context assembly for every step. Not just the input/output — the *decisions*: what sections were included, what was rejected (and why), how the token budget was spent, and how context drifted step over step.

## Integrations

### OpenAI

```python
from glassbox import wrap
from openai import OpenAI

client = wrap(OpenAI())
# All chat.completions.create() calls are now captured
```

### Anthropic

```python
from glassbox import wrap
import anthropic

client = wrap(anthropic.Anthropic())
# All messages.create() calls are now captured
```

### LangGraph

```python
from glassbox import observe

graph = observe(build_my_agent())
result = graph.invoke({"input": "resolve this case"})
# Every LLM call in every node — captured with delegation links
```

### Transparent Proxy (Zero Code Changes)

Intercept any LLM traffic without modifying your application. The proxy auto-detects Anthropic, OpenAI, and Ollama request formats on the same port.

```bash
# Start proxy + viewer in one command
glassbox proxy --provider ollama --port 4050 --viewer-port 4100
```

Or from Python:

```python
from glassbox import proxy

proxy(provider="ollama", proxy_port=4050, viewer_port=4100, working_dir=".")
```

Point your client at the proxy instead of the real API:

```python
from openai import OpenAI

# Instead of connecting to Ollama directly, go through Glassbox
client = OpenAI(base_url="http://localhost:4050/v1", api_key="unused")
response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Hello"}]
)
```

Works with any provider:

| Provider | Proxy command | Client base URL |
|----------|--------------|-----------------|
| Ollama | `glassbox proxy --provider ollama` | `http://localhost:4050/v1` |
| OpenAI | `glassbox proxy --provider openai` | `http://localhost:4050/v1` |
| Anthropic | `glassbox proxy --provider anthropic` | `http://localhost:4050` |

Every API call appears in the viewer at `http://localhost:4100` in real time.

## What Gets Captured

Each **ContextPack** records:

| Layer | What you see |
|-------|-------------|
| **Sections** | Every piece of context — system prompt, user message, tool results, retrieval chunks, memory, instructions. 8 section types. |
| **Token Budget** | How many tokens each section used. What got rejected and why (budget exceeded, relevance filtered, TTL expired, duplicate, policy blocked, dependency missing, manual exclusion). |
| **Model** | Provider, model name, temperature, max_tokens, top_p, stop sequences. |
| **Output** | Generated text, tool calls with arguments, stop reason. |
| **Metrics** | Latency, input/output tokens, cache hits, estimated cost (40+ models priced). |
| **Multi-Agent** | Parent run, parent step, delegation scope, inherited sections — full lineage across agent handoffs. |

## Features

### Context Assembler
Budget-aware context builder with automatic rejection tracking:

```python
from glassbox import ContextAssembler

assembler = ContextAssembler(budget=4000)
assembler.add(system_prompt_section)
assembler.add(retrieval_section)      # auto-rejected if budget exceeded
budget = assembler.build_budget()     # includes rejection ledger
```

### Redaction Policies
Ship ContextPacks without leaking sensitive data:

```python
from glassbox import set_redaction_policy, RedactionPolicy

set_redaction_policy(RedactionPolicy.HASH)       # SHA-256 all content
set_redaction_policy(RedactionPolicy.TRUNCATE)    # First 50 chars only
set_redaction_policy(RedactionPolicy.DROP_CONTENT) # Structure without content
```

### Context Drift Detection
Track how context evolves across steps:

```python
from glassbox import diff_context_packs

diff = diff_context_packs(step_1_pack, step_2_pack)
# Shows added, removed, and modified sections between steps
```

### Source Discovery

Glassbox scans your working directory to build a **Source Inventory** — everything that was *available* to the context assembly process, not just what made it into the LLM call.

```python
from glassbox import discover_sources

inventory = discover_sources(working_dir=".")
print(f"{inventory.total_sources} sources, {inventory.total_tokens_available:,} tokens available")
```

Automatically detects 15+ context source patterns:

| Pattern | Type |
|---------|------|
| `SKILL.md` | Skill (Claude) |
| `CLAUDE.md`, `.claude/settings.json` | Agent config |
| `.cursorrules`, `.cursor/rules/` | Agent config (Cursor) |
| `.clinerules`, `.windsurfrules` | Agent config |
| `.github/copilot-instructions.md` | Agent config (Copilot) |
| `.mcp.json`, `mcp.json` | Tool definition (MCP) |
| `system_prompt.txt`, `system_prompt.md` | Agent config |
| `.env`, `.env.local` | Environment |
| `.py`, `.ts`, `.go`, etc. | Code files |
| `.md`, `.txt`, `.rst` | Documentation |

Respects `.glassboxignore` for excluding paths. Hashes content (SHA-256) for change detection across steps.

Each source tracks a status: **included** (sent to LLM), **considered** (evaluated but not sent), **available** (on disk), **excluded** (filtered out), or **stale** (outdated version used).

This is the **pre-assembly layer** — Glassbox's two-layer context model:

- **Layer 1: Source Inventory** — What was available to the context builder
- **Layer 2: Wire Context** — What was actually sent to the LLM

The gap between these two layers is where hallucinations, stale context, and assembly blind spots live.

### Cost Estimation
Built-in pricing for 40+ models:

```python
from glassbox import estimate_cost

cost = estimate_cost("gpt-4o", input_tokens=1500, output_tokens=500)
```

### Conformance Levels
Three levels of observability depth:

- **L1 Capture** — Basic fields: sections, model, output, metrics
- **L2 Budget** — Rejection ledger populated with exclusion reasons
- **L3 Multi-Agent** — Delegation links across agent handoffs

## ContextPack Format

ContextPack is an open JSON format. Here's what one looks like:

```json
{
  "format_version": "0.1.0",
  "run_id": "run_abc123",
  "step_id": "step_0",
  "step_index": 0,
  "sections": [
    {
      "type": "system_prompt",
      "section_id": "sec_001",
      "source": "system",
      "token_count": 150,
      "content": "You are a customer support agent..."
    },
    {
      "type": "retrieval",
      "section_id": "sec_002",
      "source": "knowledge_base",
      "token_count": 800,
      "content": "Order #12345 was shipped on..."
    }
  ],
  "token_budget": {
    "total_budget": 4096,
    "total_used": 950,
    "rejected": [
      {
        "section_id": "sec_003",
        "section_type": "retrieval",
        "token_count": 3500,
        "reason": "budget_exceeded"
      }
    ]
  },
  "model": {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7
  },
  "output": {
    "type": "text",
    "text": "I can see your order #12345 was shipped...",
    "stop_reason": "stop"
  },
  "metrics": {
    "latency_ms": 1250,
    "input_tokens": 950,
    "output_tokens": 85,
    "cost_estimate_usd": 0.0034
  }
}
```

The full spec is in [`spec/CONTEXTPACK_SPEC.md`](spec/CONTEXTPACK_SPEC.md) with a JSON Schema at [`spec/context-pack.schema.json`](spec/context-pack.schema.json).

## Installation

```bash
# Just OpenAI
pip install glassbox-ctx[openai]

# Just Anthropic
pip install glassbox-ctx[anthropic]

# LangGraph agents
pip install glassbox-ctx[langgraph]

# With the viewer UI
pip install glassbox-ctx[openai,server]

# Everything
pip install glassbox-ctx[all]
```

Requires Python 3.10+.

## Viewer

```bash
glassbox serve
# or
python -c "from glassbox import serve; serve()"
```

Opens a local UI at `http://localhost:4100` showing all captured runs and steps. Three-column layout: run timeline, step detail with tabbed views (Context, Budget, Drift, Sources, Output, Model, Raw JSON), and a metadata sidebar.

### Viewer Tabs

| Tab | What it shows |
|-----|---------------|
| **Context** | Every section in the context window, color-coded by type (system prompt, user message, tool result, retrieval, memory, instruction) |
| **Budget** | Token allocation per section, rejected candidates with rejection reasons |
| **Drift** | Visual diff between consecutive steps — what was added, removed, or modified |
| **Sources** | Pre-assembly inventory: all upstream sources, inclusion status, compression ratio, source-level drift with insight metrics |
| **Output** | Model response including tool calls with structured display |
| **Model** | Provider, model name, temperature, parameters |
| **Raw** | Full ContextPack JSON |

## Real-Time Testing

### With Ollama

**Terminal 1** — Start proxy + viewer:

```bash
cd your-project/
pip install glassbox-ctx[all] openai
python -c "from glassbox import proxy; proxy(provider='ollama', proxy_port=4050, viewer_port=4100, working_dir='.')"
```

**Terminal 2** — Chat through the proxy:

```bash
python -c "from openai import OpenAI;client=OpenAI(base_url='http://localhost:4050/v1',api_key='unused');msgs=[];exec(\"while True:\n q=input('You: ')\n if q.lower()=='quit':break\n msgs.append({'role':'user','content':q})\n r=client.chat.completions.create(model='llama3.2',messages=msgs)\n print(f'\\n{r.choices[0].message.content}\\n')\n msgs.append({'role':'assistant','content':r.choices[0].message.content})\")"
```

Open `http://localhost:4100` — every message appears live.

### With Anthropic (Claude)

```bash
# Terminal 1
python -c "from glassbox import proxy; proxy(provider='anthropic', proxy_port=4050, viewer_port=4100, working_dir='.')"

# Terminal 2
export ANTHROPIC_API_KEY=your-key
python -c "from anthropic import Anthropic;c=Anthropic(base_url='http://localhost:4050');msgs=[];exec(\"while True:\n q=input('You: ')\n if q=='quit':break\n msgs.append({'role':'user','content':q})\n r=c.messages.create(model='claude-sonnet-4-20250514',max_tokens=1024,messages=msgs)\n print(f'\\n{r.content[0].text}\\n')\n msgs.append({'role':'assistant','content':r.content[0].text})\")"
```

### With OpenAI

```bash
# Terminal 1
python -c "from glassbox import proxy; proxy(provider='openai', proxy_port=4050, viewer_port=4100, working_dir='.')"

# Terminal 2
export OPENAI_API_KEY=your-key
python -c "from openai import OpenAI;client=OpenAI(base_url='http://localhost:4050/v1');msgs=[];exec(\"while True:\n q=input('You: ')\n if q=='quit':break\n msgs.append({'role':'user','content':q})\n r=client.chat.completions.create(model='gpt-4o',messages=msgs)\n print(f'\\n{r.choices[0].message.content}\\n')\n msgs.append({'role':'assistant','content':r.choices[0].message.content})\")"
```

## CLI Reference

```bash
# Start proxy (captures LLM traffic) + viewer
glassbox proxy --provider <ollama|openai|anthropic> [--port 4050] [--viewer-port 4100] [--working-dir .]

# Start viewer only (browse previously captured runs)
glassbox serve [--port 4100] [--storage-dir ~/.glassbox]

# List captured runs
glassbox ls

# Inspect a specific run
glassbox inspect <run-id>
```

## Development

```bash
git clone https://github.com/arq-ai-labs/glassbox.git
cd glassbox
pip install -e ".[dev]"
pytest tests/ -v
```

## License

[Apache 2.0](LICENSE)
