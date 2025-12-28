# Debug Observability System

## Overview

A structured observability and replay system for debugging bicker-bot's decision pipeline. Three main capabilities:

1. **Decision tracing** — Structured records of what happened and why at each pipeline stage
2. **Web debugger** — Browse traces, replay with config modifications, A/B comparison
3. **Memory browser** — Search, view, and delete memories from ChromaDB

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR                            │
│  - Creates TraceContext per message                         │
│  - Passes context through pipeline                          │
│  - Stores completed traces                                  │
│  - Handles replay requests                                  │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │ TraceStore  │     │ConfigLoader │     │DebugServer  │
    │  (SQLite)   │     │  (files)    │     │  (FastAPI)  │
    └─────────────┘     └─────────────┘     └─────────────┘
```

## Data Model

### TraceStep

Captures a single pipeline stage:

```python
@dataclass
class TraceStep:
    stage: str              # "gate", "engagement", "selector", "context", "responder"
    timestamp: datetime
    inputs: dict            # What this stage received
    outputs: dict           # What it produced
    decision: str           # Human-readable summary
    details: dict           # Stage-specific data

    # LLM-specific fields (None for non-LLM stages)
    model: str | None
    prompt: str | None
    raw_response: str | None
    thinking: str | None             # Claude's thinking blocks
    thought_signatures: list[str] | None  # Gemini's thought signatures
    token_usage: dict | None         # {"input": N, "output": N, "thinking": N}
```

### TraceContext

Flows through the pipeline, accumulating steps:

```python
@dataclass
class TraceContext:
    id: str                          # UUID
    started_at: datetime
    channel: str
    trigger_messages: list[str]      # IRC messages that triggered this
    config_snapshot: dict            # Frozen config at trace start
    steps: list[TraceStep]
    final_result: list[str] | None   # Messages sent (or None if declined)
    is_replay: bool
    original_trace_id: str | None    # If replay, points to original
```

### Usage in Pipeline

Each component appends its step:

```python
# Example: gate.py
def should_respond(self, messages, ctx: TraceContext) -> bool:
    prob = self._calculate_probability(messages)
    result = random.random() < prob

    ctx.add_step("gate",
        inputs={"messages": [m.text for m in messages]},
        outputs={"probability": prob, "responded": result},
        decision=f"{'Responded' if result else 'Declined'}: P={prob:.2f}")

    return result
```

## Config System

### File Structure

```
config/
├── prompts/
│   ├── merry.md           # Merry's personality/system prompt
│   ├── hachiman.md        # Hachiman's personality/system prompt
│   ├── engagement.md      # Engagement check prompt
│   └── context.md         # Context builder system prompt
└── policies.yaml          # Non-prompt settings
```

### Prompt Format

Markdown with YAML frontmatter:

```markdown
---
model: gemini-3-pro-preview
max_tokens: 1024
temperature: 0.9
thinking: false
---

You are Merry Nightmare, a dream demon from the anime "Yumekui Merry"...
```

### Policies

```yaml
gate:
  base_prob: 0.05
  mention_prob: 0.8
  decay_factor: 0.5

context:
  max_tool_rounds: 3
  rag_top_k: 5

memory:
  extraction_enabled: true
  high_intensity_threshold: 0.7
```

### ConfigLoader

```python
class ConfigLoader:
    def __init__(self, config_dir: Path): ...
    def reload(self): ...              # Hot-reload from disk
    def snapshot(self) -> dict: ...    # Frozen copy for traces
```

## Trace Storage

SQLite with JSON blob for full trace:

```sql
CREATE TABLE traces (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    channel TEXT NOT NULL,
    bot TEXT,
    trigger_text TEXT,
    outcome TEXT NOT NULL,
    is_replay INTEGER DEFAULT 0,
    original_trace_id TEXT,
    trace_json TEXT NOT NULL
);

CREATE INDEX idx_traces_created ON traces(created_at DESC);
CREATE INDEX idx_traces_channel ON traces(channel);
CREATE INDEX idx_traces_bot ON traces(bot);
```

### TraceStore API

```python
class TraceStore:
    def save(self, trace: TraceContext) -> None: ...
    def get(self, trace_id: str) -> TraceContext | None: ...
    def recent(self, limit: int = 50, channel: str = None) -> list[TraceSummary]: ...
    def prune(self, keep_last: int = 500) -> int: ...
```

## Web UI

FastAPI + htmx + Jinja2, same process as bot.

### Routes

```
/                     → Redirect to /traces
/traces               → List of recent traces
/traces/{id}          → Single trace detail view
/traces/{id}/replay   → Trigger replay, returns comparison view
/memories             → Memory browser
/memories/search      → htmx endpoint for search results
/memories/{id}/delete → Delete a memory
/config/modal         → Config editor modal (htmx)
/config/reload        → Trigger hot-reload
```

### Views

**Trace list** (`/traces`):
- Table: timestamp, channel, bot, trigger preview, outcome
- Filters for channel/bot/outcome
- Click row to expand detail

**Trace detail** (`/traces/{id}`):
- Collapsible sections per pipeline stage
- Each shows: inputs → decision → outputs
- LLM stages: expandable prompt/thinking/raw response
- "Replay" button at top

**Replay comparison**:
- Side-by-side: original vs replayed
- Diff highlighting where decisions diverged
- Config diff shown

**Memory browser** (`/memories`):
- Collection dropdown + search box
- Results: preview, collection, similarity, date
- Click to expand, delete button per row

**Config modal**:
- Tabs for each prompt + policies
- Session overrides (don't affect live bot)
- "Reload from disk" / "Clear overrides" / "Save to disk" actions
- Badge shows active override count

### Session Overrides

```python
@dataclass
class ConfigOverride:
    path: str      # e.g., "prompts/merry.md"
    value: str     # New content

class DebugSession:
    overrides: dict[str, ConfigOverride]

    def effective_config(self, base: dict) -> dict:
        """Apply overrides on top of base config."""
```

Overrides stored in browser localStorage. Replays use effective config (base + overrides).

## Replay Mechanism

```python
@dataclass
class ReplayResult:
    original: TraceContext
    replayed: TraceContext
    config_diff: dict
    decision_diffs: list[DecisionDiff]

@dataclass
class DecisionDiff:
    stage: str
    original_decision: str
    replayed_decision: str
    diverged: bool

class Orchestrator:
    async def replay(
        self,
        trace_id: str,
        config_overrides: dict | None = None
    ) -> ReplayResult:
        original = self.trace_store.get(trace_id)

        # Build effective config
        effective = apply_overrides(
            original.config_snapshot,
            config_overrides or {}
        )

        # Create replay context
        ctx = TraceContext(
            trigger_messages=original.trigger_messages,
            config_snapshot=effective,
            is_replay=True,
            original_trace_id=original.id
        )

        # Run pipeline in dry-run mode (no IRC, no memory writes)
        await self._run_pipeline(
            messages=original.trigger_messages,
            ctx=ctx,
            dry_run=True
        )

        self.trace_store.save(ctx)
        return ReplayResult(original, ctx, ...)
```

## Memory Browser

```python
@dataclass
class MemorySearchResult:
    id: str
    collection: str
    text: str
    metadata: dict
    similarity: float

class MemoryBrowser:
    def collections(self) -> list[str]: ...
    def search(self, query: str, collection: str = None, limit: int = 20) -> list[MemorySearchResult]: ...
    def browse(self, collection: str, offset: int = 0, limit: int = 50) -> list[MemorySearchResult]: ...
    def delete(self, collection: str, memory_id: str) -> bool: ...
```

## File Structure

```
src/bicker_bot/
├── debug/
│   ├── __init__.py
│   ├── server.py           # FastAPI app, routes
│   ├── templates/
│   │   ├── base.html
│   │   ├── traces.html
│   │   ├── trace_detail.html
│   │   ├── memories.html
│   │   └── partials/
│   │       ├── trace_list.html
│   │       ├── trace_step.html
│   │       ├── config_modal.html
│   │       ├── replay_comparison.html
│   │       └── memory_results.html
│   └── static/
│       └── style.css
├── tracing/
│   ├── __init__.py
│   ├── context.py          # TraceContext, TraceStep
│   └── store.py            # TraceStore (SQLite)
├── config/
│   ├── __init__.py
│   └── loader.py           # ConfigLoader
└── ...

config/                     # Outside src/
├── prompts/
│   ├── merry.md
│   ├── hachiman.md
│   ├── engagement.md
│   └── context.md
└── policies.yaml

data/
└── traces.db
```

## Integration

Debug server runs in same process as bot:

```python
# orchestrator.py
async def run(self):
    debug_server = DebugServer(
        trace_store=self.trace_store,
        config_loader=self.config_loader,
        memory_store=self.memory_store,
        replay_fn=self._replay_pipeline
    )
    asyncio.create_task(debug_server.serve(port=8080))
    await self.irc_client.run_forever()
```

Access via `ssh -L 8080:localhost:8080 server`.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Tracing approach | TraceContext passed through | Explicit, type-safe, scoped |
| Storage | SQLite + filesystem | Traces queryable, configs editable |
| Prompt format | Markdown + frontmatter | Readable, editor-friendly |
| Web framework | FastAPI + htmx | Testable, no JS framework |
| Config overrides | Session-based | Experiment without affecting live bot |
| Replay scope | Full pipeline, A/B comparison | Direct "did this help?" feedback |
| Process model | Same process | Simplicity, direct access |
