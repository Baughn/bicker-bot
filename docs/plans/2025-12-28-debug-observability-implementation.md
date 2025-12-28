# Debug Observability System - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a structured observability system with web UI for debugging the bicker-bot decision pipeline.

**Architecture:** TraceContext flows through pipeline stages, each appending TraceSteps. Traces stored in SQLite. FastAPI+htmx web UI for browsing/replaying. Config loaded from markdown+frontmatter files with session-based overrides.

**Tech Stack:** FastAPI, Jinja2, htmx, SQLite, python-frontmatter

---

## Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add new dependencies**

Add to `dependencies` in pyproject.toml:
```toml
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "python-frontmatter>=1.1.0",
```

**Step 2: Sync dependencies**

Run: `uv sync`
Expected: Dependencies installed successfully

**Step 3: Commit**

```bash
jj commit -m "Add FastAPI and frontmatter dependencies for debug server"
```

---

## Task 2: TraceStep Data Model

**Files:**
- Create: `src/bicker_bot/tracing/__init__.py`
- Create: `src/bicker_bot/tracing/context.py`
- Test: `tests/test_tracing.py`

**Step 1: Write the failing test**

Create `tests/test_tracing.py`:
```python
"""Tests for tracing data model."""

from datetime import datetime

import pytest

from bicker_bot.tracing.context import TraceStep


class TestTraceStep:
    """Tests for TraceStep dataclass."""

    def test_create_basic_step(self):
        """Test creating a basic trace step."""
        step = TraceStep(
            stage="gate",
            inputs={"message": "hello"},
            outputs={"probability": 0.5},
            decision="Declined: P=0.5",
        )
        assert step.stage == "gate"
        assert step.inputs == {"message": "hello"}
        assert step.outputs == {"probability": 0.5}
        assert step.decision == "Declined: P=0.5"
        assert step.timestamp is not None

    def test_llm_fields_optional(self):
        """Test that LLM fields are optional."""
        step = TraceStep(
            stage="gate",
            inputs={},
            outputs={},
            decision="test",
        )
        assert step.model is None
        assert step.prompt is None
        assert step.raw_response is None
        assert step.thinking is None
        assert step.thought_signatures is None
        assert step.token_usage is None

    def test_llm_fields_populated(self):
        """Test creating step with LLM fields."""
        step = TraceStep(
            stage="engagement",
            inputs={"message": "test"},
            outputs={"probability": 0.8},
            decision="Engaged",
            model="gemini-3-flash-preview",
            prompt="Is this engaging?",
            raw_response='{"probability": 80}',
            thinking=None,
            thought_signatures=["sig1", "sig2"],
            token_usage={"input": 100, "output": 50},
        )
        assert step.model == "gemini-3-flash-preview"
        assert step.thought_signatures == ["sig1", "sig2"]

    def test_to_dict(self):
        """Test serialization to dict."""
        step = TraceStep(
            stage="gate",
            inputs={"x": 1},
            outputs={"y": 2},
            decision="test",
        )
        d = step.to_dict()
        assert d["stage"] == "gate"
        assert d["inputs"] == {"x": 1}
        assert "timestamp" in d

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "stage": "responder",
            "timestamp": "2025-01-01T12:00:00",
            "inputs": {"msg": "hi"},
            "outputs": {"reply": "hello"},
            "decision": "responded",
            "details": {},
            "model": "claude-opus-4-5-20251101",
            "prompt": "test prompt",
            "raw_response": "test response",
            "thinking": "thinking block",
            "thought_signatures": None,
            "token_usage": {"input": 200, "output": 100},
        }
        step = TraceStep.from_dict(data)
        assert step.stage == "responder"
        assert step.model == "claude-opus-4-5-20251101"
        assert step.thinking == "thinking block"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: FAIL with ModuleNotFoundError (tracing module doesn't exist)

**Step 3: Create the module structure**

Create `src/bicker_bot/tracing/__init__.py`:
```python
"""Tracing module for debug observability."""

from bicker_bot.tracing.context import TraceContext, TraceStep

__all__ = ["TraceContext", "TraceStep"]
```

**Step 4: Write minimal implementation**

Create `src/bicker_bot/tracing/context.py`:
```python
"""Trace context and step data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TraceStep:
    """A single step in the pipeline trace."""

    stage: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    decision: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    # LLM-specific fields (None for non-LLM stages)
    model: str | None = None
    prompt: str | None = None
    raw_response: str | None = None
    thinking: str | None = None
    thought_signatures: list[str] | None = None
    token_usage: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "stage": self.stage,
            "timestamp": self.timestamp.isoformat(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "decision": self.decision,
            "details": self.details,
            "model": self.model,
            "prompt": self.prompt,
            "raw_response": self.raw_response,
            "thinking": self.thinking,
            "thought_signatures": self.thought_signatures,
            "token_usage": self.token_usage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceStep":
        """Deserialize from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            stage=data["stage"],
            timestamp=timestamp,
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            decision=data.get("decision", ""),
            details=data.get("details", {}),
            model=data.get("model"),
            prompt=data.get("prompt"),
            raw_response=data.get("raw_response"),
            thinking=data.get("thinking"),
            thought_signatures=data.get("thought_signatures"),
            token_usage=data.get("token_usage"),
        )
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: PASS

**Step 6: Commit**

```bash
jj commit -m "Add TraceStep data model for pipeline tracing"
```

---

## Task 3: TraceContext Data Model

**Files:**
- Modify: `src/bicker_bot/tracing/context.py`
- Modify: `tests/test_tracing.py`

**Step 1: Add tests for TraceContext**

Append to `tests/test_tracing.py`:
```python
from bicker_bot.tracing.context import TraceContext


class TestTraceContext:
    """Tests for TraceContext."""

    def test_create_context(self):
        """Test creating a trace context."""
        ctx = TraceContext(
            channel="#test",
            trigger_messages=["hello world"],
            config_snapshot={"gate": {"base_prob": 0.05}},
        )
        assert ctx.id is not None
        assert ctx.channel == "#test"
        assert ctx.trigger_messages == ["hello world"]
        assert ctx.steps == []
        assert ctx.is_replay is False

    def test_add_step(self):
        """Test adding steps to context."""
        ctx = TraceContext(
            channel="#test",
            trigger_messages=["test"],
            config_snapshot={},
        )
        ctx.add_step(
            stage="gate",
            inputs={"msg": "test"},
            outputs={"prob": 0.5},
            decision="declined",
        )
        assert len(ctx.steps) == 1
        assert ctx.steps[0].stage == "gate"

    def test_add_llm_step(self):
        """Test adding LLM step with extra fields."""
        ctx = TraceContext(
            channel="#test",
            trigger_messages=["test"],
            config_snapshot={},
        )
        ctx.add_llm_step(
            stage="engagement",
            inputs={"msg": "test"},
            outputs={"prob": 0.8},
            decision="engaged",
            model="gemini-3-flash-preview",
            prompt="Is this engaging?",
            raw_response='{"probability": 80}',
            thinking=None,
            thought_signatures=["sig1"],
            token_usage={"input": 100, "output": 50},
        )
        assert len(ctx.steps) == 1
        step = ctx.steps[0]
        assert step.model == "gemini-3-flash-preview"
        assert step.thought_signatures == ["sig1"]

    def test_to_dict_and_back(self):
        """Test round-trip serialization."""
        ctx = TraceContext(
            channel="#test",
            trigger_messages=["hello"],
            config_snapshot={"key": "value"},
        )
        ctx.add_step("gate", {"a": 1}, {"b": 2}, "test")

        d = ctx.to_dict()
        restored = TraceContext.from_dict(d)

        assert restored.id == ctx.id
        assert restored.channel == ctx.channel
        assert len(restored.steps) == 1
        assert restored.steps[0].stage == "gate"

    def test_replay_context(self):
        """Test creating a replay context."""
        ctx = TraceContext(
            channel="#test",
            trigger_messages=["original"],
            config_snapshot={},
            is_replay=True,
            original_trace_id="abc123",
        )
        assert ctx.is_replay is True
        assert ctx.original_trace_id == "abc123"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing.py::TestTraceContext -v`
Expected: FAIL (TraceContext not implemented)

**Step 3: Implement TraceContext**

Add to `src/bicker_bot/tracing/context.py`:
```python
from uuid import uuid4


@dataclass
class TraceContext:
    """Context that flows through the pipeline, accumulating trace steps."""

    channel: str
    trigger_messages: list[str]
    config_snapshot: dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid4()))
    started_at: datetime = field(default_factory=datetime.now)
    steps: list[TraceStep] = field(default_factory=list)
    final_result: list[str] | None = None
    is_replay: bool = False
    original_trace_id: str | None = None

    def add_step(
        self,
        stage: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        decision: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Add a non-LLM step to the trace."""
        self.steps.append(
            TraceStep(
                stage=stage,
                inputs=inputs,
                outputs=outputs,
                decision=decision,
                details=details or {},
            )
        )

    def add_llm_step(
        self,
        stage: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        decision: str,
        model: str,
        prompt: str,
        raw_response: str,
        thinking: str | None = None,
        thought_signatures: list[str] | None = None,
        token_usage: dict[str, int] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Add an LLM step to the trace."""
        self.steps.append(
            TraceStep(
                stage=stage,
                inputs=inputs,
                outputs=outputs,
                decision=decision,
                details=details or {},
                model=model,
                prompt=prompt,
                raw_response=raw_response,
                thinking=thinking,
                thought_signatures=thought_signatures,
                token_usage=token_usage,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat(),
            "channel": self.channel,
            "trigger_messages": self.trigger_messages,
            "config_snapshot": self.config_snapshot,
            "steps": [step.to_dict() for step in self.steps],
            "final_result": self.final_result,
            "is_replay": self.is_replay,
            "original_trace_id": self.original_trace_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceContext":
        """Deserialize from dictionary."""
        started_at = data.get("started_at")
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)
        elif started_at is None:
            started_at = datetime.now()

        ctx = cls(
            id=data["id"],
            started_at=started_at,
            channel=data["channel"],
            trigger_messages=data.get("trigger_messages", []),
            config_snapshot=data.get("config_snapshot", {}),
            final_result=data.get("final_result"),
            is_replay=data.get("is_replay", False),
            original_trace_id=data.get("original_trace_id"),
        )
        ctx.steps = [TraceStep.from_dict(s) for s in data.get("steps", [])]
        return ctx
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: PASS

**Step 5: Commit**

```bash
jj commit -m "Add TraceContext for accumulating pipeline trace steps"
```

---

## Task 4: TraceStore (SQLite Storage)

**Files:**
- Create: `src/bicker_bot/tracing/store.py`
- Modify: `src/bicker_bot/tracing/__init__.py`
- Modify: `tests/test_tracing.py`

**Step 1: Add tests for TraceStore**

Append to `tests/test_tracing.py`:
```python
from pathlib import Path

from bicker_bot.tracing.store import TraceStore


class TestTraceStore:
    """Tests for TraceStore."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> TraceStore:
        """Create a temporary trace store."""
        db_path = tmp_path / "traces.db"
        return TraceStore(db_path)

    def test_save_and_get(self, store: TraceStore):
        """Test saving and retrieving a trace."""
        ctx = TraceContext(
            channel="#test",
            trigger_messages=["hello"],
            config_snapshot={"key": "value"},
        )
        ctx.add_step("gate", {"a": 1}, {"b": 2}, "test decision")
        ctx.final_result = ["response"]

        store.save(ctx)
        retrieved = store.get(ctx.id)

        assert retrieved is not None
        assert retrieved.id == ctx.id
        assert retrieved.channel == "#test"
        assert len(retrieved.steps) == 1

    def test_get_nonexistent(self, store: TraceStore):
        """Test getting a trace that doesn't exist."""
        result = store.get("nonexistent-id")
        assert result is None

    def test_recent(self, store: TraceStore):
        """Test getting recent traces."""
        for i in range(5):
            ctx = TraceContext(
                channel="#test",
                trigger_messages=[f"msg{i}"],
                config_snapshot={},
            )
            ctx.final_result = [f"response{i}"]
            store.save(ctx)

        recent = store.recent(limit=3)
        assert len(recent) == 3

    def test_recent_filter_by_channel(self, store: TraceStore):
        """Test filtering recent traces by channel."""
        for channel in ["#a", "#a", "#b"]:
            ctx = TraceContext(
                channel=channel,
                trigger_messages=["test"],
                config_snapshot={},
            )
            store.save(ctx)

        recent = store.recent(channel="#a")
        assert len(recent) == 2
        assert all(t.channel == "#a" for t in recent)

    def test_prune(self, store: TraceStore):
        """Test pruning old traces."""
        for i in range(10):
            ctx = TraceContext(
                channel="#test",
                trigger_messages=[f"msg{i}"],
                config_snapshot={},
            )
            store.save(ctx)

        deleted = store.prune(keep_last=3)
        assert deleted == 7
        assert len(store.recent(limit=100)) == 3
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing.py::TestTraceStore -v`
Expected: FAIL (TraceStore not implemented)

**Step 3: Implement TraceStore**

Create `src/bicker_bot/tracing/store.py`:
```python
"""SQLite storage for traces."""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bicker_bot.tracing.context import TraceContext

logger = logging.getLogger(__name__)


@dataclass
class TraceSummary:
    """Lightweight summary of a trace for list views."""

    id: str
    created_at: datetime
    channel: str
    bot: str | None
    trigger_text: str
    outcome: str
    is_replay: bool


class TraceStore:
    """SQLite-backed trace storage."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()
        logger.info(f"TraceStore initialized at {self.db_path}")

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS traces (
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
            CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_traces_channel ON traces(channel);
            CREATE INDEX IF NOT EXISTS idx_traces_bot ON traces(bot);
        """)
        self._conn.commit()

    def save(self, trace: TraceContext) -> None:
        """Save a trace to the database."""
        # Determine outcome based on steps and final_result
        if trace.final_result:
            outcome = "responded"
        elif any(s.stage == "engagement" for s in trace.steps):
            outcome = "declined_engagement"
        else:
            outcome = "declined_gate"

        # Determine which bot responded (if any)
        bot = None
        for step in trace.steps:
            if step.stage == "selector" and "selected" in step.outputs:
                bot = step.outputs["selected"]
                break
            if step.stage == "responder" and step.model:
                # Infer from model
                if "claude" in step.model.lower():
                    bot = "hachiman"
                elif "gemini" in step.model.lower():
                    bot = "merry"
                break

        # Get first trigger message for preview
        trigger_text = trace.trigger_messages[0][:100] if trace.trigger_messages else ""

        self._conn.execute(
            """
            INSERT OR REPLACE INTO traces
            (id, created_at, channel, bot, trigger_text, outcome, is_replay, original_trace_id, trace_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.id,
                trace.started_at.isoformat(),
                trace.channel,
                bot,
                trigger_text,
                outcome,
                1 if trace.is_replay else 0,
                trace.original_trace_id,
                json.dumps(trace.to_dict()),
            ),
        )
        self._conn.commit()
        logger.debug(f"Saved trace {trace.id[:8]}... outcome={outcome}")

    def get(self, trace_id: str) -> TraceContext | None:
        """Get a trace by ID."""
        row = self._conn.execute(
            "SELECT trace_json FROM traces WHERE id = ?", (trace_id,)
        ).fetchone()
        if row is None:
            return None
        return TraceContext.from_dict(json.loads(row["trace_json"]))

    def recent(
        self,
        limit: int = 50,
        channel: str | None = None,
        bot: str | None = None,
    ) -> list[TraceSummary]:
        """Get recent trace summaries."""
        query = "SELECT id, created_at, channel, bot, trigger_text, outcome, is_replay FROM traces"
        params: list = []
        conditions = []

        if channel:
            conditions.append("channel = ?")
            params.append(channel)
        if bot:
            conditions.append("bot = ?")
            params.append(bot)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [
            TraceSummary(
                id=row["id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                channel=row["channel"],
                bot=row["bot"],
                trigger_text=row["trigger_text"] or "",
                outcome=row["outcome"],
                is_replay=bool(row["is_replay"]),
            )
            for row in rows
        ]

    def prune(self, keep_last: int = 500) -> int:
        """Delete old traces, keeping the most recent. Returns count deleted."""
        # Get the ID of the Nth most recent trace
        cutoff_row = self._conn.execute(
            "SELECT id FROM traces ORDER BY created_at DESC LIMIT 1 OFFSET ?",
            (keep_last - 1,),
        ).fetchone()

        if cutoff_row is None:
            return 0  # Fewer traces than keep_last

        # Get the created_at of that trace
        cutoff = self._conn.execute(
            "SELECT created_at FROM traces WHERE id = ?", (cutoff_row["id"],)
        ).fetchone()

        if cutoff is None:
            return 0

        # Delete older traces
        result = self._conn.execute(
            "DELETE FROM traces WHERE created_at < ?", (cutoff["created_at"],)
        )
        self._conn.commit()
        deleted = result.rowcount
        if deleted > 0:
            logger.info(f"Pruned {deleted} old traces")
        return deleted

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
```

**Step 4: Update __init__.py**

Update `src/bicker_bot/tracing/__init__.py`:
```python
"""Tracing module for debug observability."""

from bicker_bot.tracing.context import TraceContext, TraceStep
from bicker_bot.tracing.store import TraceStore, TraceSummary

__all__ = ["TraceContext", "TraceStep", "TraceStore", "TraceSummary"]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: PASS

**Step 6: Commit**

```bash
jj commit -m "Add TraceStore for SQLite-backed trace persistence"
```

---

## Task 5: ConfigLoader - File Structure and Parsing

**Files:**
- Create: `src/bicker_bot/debug/__init__.py`
- Create: `src/bicker_bot/debug/config_loader.py`
- Create: `tests/test_debug_config.py`

**Step 1: Write the failing test**

Create `tests/test_debug_config.py`:
```python
"""Tests for debug config loader."""

from pathlib import Path

import pytest

from bicker_bot.debug.config_loader import ConfigLoader, PromptConfig


class TestConfigLoader:
    """Tests for ConfigLoader."""

    @pytest.fixture
    def config_dir(self, tmp_path: Path) -> Path:
        """Create a config directory with test files."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        # Create a test prompt file
        (prompts_dir / "merry.md").write_text("""---
model: gemini-3-pro-preview
max_tokens: 1024
temperature: 0.9
---

You are Merry Nightmare, a dream demon.
Direct and action-oriented.
""")

        # Create policies file
        (tmp_path / "policies.yaml").write_text("""
gate:
  base_prob: 0.05
  decay_factor: 0.5

context:
  max_tool_rounds: 3
""")

        return tmp_path

    def test_load_prompt(self, config_dir: Path):
        """Test loading a prompt file."""
        loader = ConfigLoader(config_dir)
        prompt = loader.get_prompt("merry")

        assert prompt is not None
        assert prompt.model == "gemini-3-pro-preview"
        assert prompt.max_tokens == 1024
        assert prompt.temperature == 0.9
        assert "Merry Nightmare" in prompt.content

    def test_load_policies(self, config_dir: Path):
        """Test loading policies file."""
        loader = ConfigLoader(config_dir)
        policies = loader.get_policies()

        assert policies["gate"]["base_prob"] == 0.05
        assert policies["context"]["max_tool_rounds"] == 3

    def test_snapshot(self, config_dir: Path):
        """Test getting a frozen snapshot."""
        loader = ConfigLoader(config_dir)
        snapshot1 = loader.snapshot()
        snapshot2 = loader.snapshot()

        # Should be equal but not the same object
        assert snapshot1 == snapshot2
        assert snapshot1 is not snapshot2

    def test_reload(self, config_dir: Path):
        """Test hot-reloading config."""
        loader = ConfigLoader(config_dir)
        original = loader.get_policies()

        # Modify the file
        (config_dir / "policies.yaml").write_text("""
gate:
  base_prob: 0.10
""")

        # Reload
        loader.reload()
        updated = loader.get_policies()

        assert original["gate"]["base_prob"] == 0.05
        assert updated["gate"]["base_prob"] == 0.10

    def test_missing_prompt(self, config_dir: Path):
        """Test getting a nonexistent prompt."""
        loader = ConfigLoader(config_dir)
        prompt = loader.get_prompt("nonexistent")
        assert prompt is None

    def test_list_prompts(self, config_dir: Path):
        """Test listing available prompts."""
        loader = ConfigLoader(config_dir)
        prompts = loader.list_prompts()
        assert "merry" in prompts
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_debug_config.py -v`
Expected: FAIL (module doesn't exist)

**Step 3: Create the module structure**

Create `src/bicker_bot/debug/__init__.py`:
```python
"""Debug server and observability tools."""

from bicker_bot.debug.config_loader import ConfigLoader, PromptConfig

__all__ = ["ConfigLoader", "PromptConfig"]
```

**Step 4: Implement ConfigLoader**

Create `src/bicker_bot/debug/config_loader.py`:
```python
"""Configuration loader for debug-time config files."""

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter
import yaml

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """Configuration for a prompt file."""

    name: str
    content: str
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    thinking: bool | None = None
    raw_frontmatter: dict[str, Any] | None = None


class ConfigLoader:
    """Loads and manages debug-time configuration files."""

    def __init__(self, config_dir: Path | str):
        self.config_dir = Path(config_dir)
        self._prompts: dict[str, PromptConfig] = {}
        self._policies: dict[str, Any] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all configuration files."""
        self._load_prompts()
        self._load_policies()

    def _load_prompts(self) -> None:
        """Load all prompt files from prompts/ directory."""
        prompts_dir = self.config_dir / "prompts"
        if not prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {prompts_dir}")
            return

        self._prompts.clear()
        for prompt_file in prompts_dir.glob("*.md"):
            try:
                post = frontmatter.load(prompt_file)
                name = prompt_file.stem

                self._prompts[name] = PromptConfig(
                    name=name,
                    content=post.content,
                    model=post.get("model"),
                    max_tokens=post.get("max_tokens"),
                    temperature=post.get("temperature"),
                    thinking=post.get("thinking"),
                    raw_frontmatter=dict(post.metadata),
                )
                logger.debug(f"Loaded prompt: {name}")
            except Exception as e:
                logger.error(f"Failed to load prompt {prompt_file}: {e}")

        logger.info(f"Loaded {len(self._prompts)} prompts")

    def _load_policies(self) -> None:
        """Load policies.yaml file."""
        policies_file = self.config_dir / "policies.yaml"
        if not policies_file.exists():
            logger.warning(f"Policies file not found: {policies_file}")
            self._policies = {}
            return

        try:
            with open(policies_file) as f:
                self._policies = yaml.safe_load(f) or {}
            logger.info(f"Loaded policies with {len(self._policies)} sections")
        except Exception as e:
            logger.error(f"Failed to load policies: {e}")
            self._policies = {}

    def reload(self) -> None:
        """Hot-reload all configuration files."""
        logger.info("Reloading configuration files")
        self._load_all()

    def get_prompt(self, name: str) -> PromptConfig | None:
        """Get a prompt configuration by name."""
        return self._prompts.get(name)

    def get_policies(self) -> dict[str, Any]:
        """Get the policies configuration."""
        return copy.deepcopy(self._policies)

    def list_prompts(self) -> list[str]:
        """List available prompt names."""
        return list(self._prompts.keys())

    def snapshot(self) -> dict[str, Any]:
        """Return a frozen copy of all configuration."""
        return {
            "prompts": {
                name: {
                    "content": p.content,
                    "model": p.model,
                    "max_tokens": p.max_tokens,
                    "temperature": p.temperature,
                    "thinking": p.thinking,
                }
                for name, p in self._prompts.items()
            },
            "policies": copy.deepcopy(self._policies),
        }
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_debug_config.py -v`
Expected: PASS

**Step 6: Commit**

```bash
jj commit -m "Add ConfigLoader for markdown+frontmatter prompt files"
```

---

## Task 6: Instrument ResponseGate with Tracing

**Files:**
- Modify: `src/bicker_bot/core/gate.py`
- Modify: `tests/test_gate.py`

**Step 1: Add test for traced gate**

Add to `tests/test_gate.py`:
```python
from bicker_bot.tracing import TraceContext


class TestGateTracing:
    """Tests for gate tracing integration."""

    def test_gate_adds_trace_step(self, gate_config: GateConfig, bot_nicks: tuple[str, str]):
        """Test that gate adds a step to trace context."""
        gate = ResponseGate(gate_config, bot_nicks)
        ctx = TraceContext(
            channel="#test",
            trigger_messages=["Hello Merry?"],
            config_snapshot={},
        )

        result = gate.should_respond(
            "Hello Merry?", None, 0, _roll=0.01, trace_ctx=ctx
        )

        assert len(ctx.steps) == 1
        step = ctx.steps[0]
        assert step.stage == "gate"
        assert "probability" in step.outputs
        assert "roll" in step.outputs
        assert step.decision != ""

    def test_gate_works_without_trace(self, gate_config: GateConfig, bot_nicks: tuple[str, str]):
        """Test that gate still works when no trace context provided."""
        gate = ResponseGate(gate_config, bot_nicks)

        # Should not raise
        result = gate.should_respond("Hello", None, 0, _roll=0.5)
        assert isinstance(result, GateResult)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_gate.py::TestGateTracing -v`
Expected: FAIL (trace_ctx parameter doesn't exist)

**Step 3: Modify ResponseGate.should_respond**

Update `src/bicker_bot/core/gate.py` - modify the `should_respond` method signature and add tracing:

```python
# Add import at top
from bicker_bot.tracing import TraceContext

# Modify should_respond method
def should_respond(
    self,
    message: str,
    last_activity: datetime | None,
    consecutive_bot_messages: int,
    current_time: datetime | None = None,
    is_mode_change: bool = False,
    _roll: float | None = None,  # For testing
    trace_ctx: TraceContext | None = None,  # Add this parameter
) -> GateResult:
    """Decide whether to respond to a message.

    Args:
        message: The message content
        last_activity: Timestamp of last channel activity
        consecutive_bot_messages: Number of consecutive bot messages
        current_time: Current time (for testing)
        is_mode_change: Whether this is a channel mode change event
        _roll: Override random roll (for testing)
        trace_ctx: Optional trace context for debugging

    Returns:
        GateResult with decision and metadata
    """
    factors = self.analyze_factors(
        message=message,
        last_activity=last_activity,
        consecutive_bot_messages=consecutive_bot_messages,
        current_time=current_time,
        is_mode_change=is_mode_change,
    )

    probability = self.calculate_probability(factors)
    roll = _roll if _roll is not None else random.random()

    result = GateResult(
        should_respond=roll < probability,
        probability=probability,
        factors=factors,
        roll=roll,
    )

    # Add trace step if context provided
    if trace_ctx is not None:
        trace_ctx.add_step(
            stage="gate",
            inputs={
                "message": message[:200],  # Truncate for storage
                "consecutive_bot_messages": consecutive_bot_messages,
                "is_mode_change": is_mode_change,
            },
            outputs={
                "probability": probability,
                "roll": roll,
                "should_respond": result.should_respond,
            },
            decision=f"{'PASS' if result.should_respond else 'FAIL'}: P={probability:.3f} roll={roll:.3f}",
            details={
                "factors": {
                    "mentioned": factors.mentioned,
                    "is_question": factors.is_question,
                    "is_conversation_start": factors.is_conversation_start,
                    "directly_addressed": factors.directly_addressed,
                    "addressed_bot": factors.addressed_bot,
                },
            },
        )

    # ... rest of existing logging code ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_gate.py -v`
Expected: PASS

**Step 5: Commit**

```bash
jj commit -m "Add tracing support to ResponseGate"
```

---

## Task 7: Instrument EngagementChecker with Tracing

**Files:**
- Modify: `src/bicker_bot/core/engagement.py`
- Create: `tests/test_engagement_tracing.py`

**Step 1: Write the failing test**

Create `tests/test_engagement_tracing.py`:
```python
"""Tests for engagement checker tracing."""

import pytest

from bicker_bot.tracing import TraceContext


class TestEngagementTracing:
    """Tests for engagement tracing integration."""

    @pytest.mark.asyncio
    async def test_engagement_adds_llm_trace_step(self, has_api_keys: bool):
        """Test that engagement checker adds LLM step to trace."""
        if not has_api_keys:
            pytest.skip("API keys not available")

        import os
        from bicker_bot.core.engagement import EngagementChecker

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        checker = EngagementChecker(api_key)

        ctx = TraceContext(
            channel="#test",
            trigger_messages=["Hello, anyone there?"],
            config_snapshot={},
        )

        result = await checker.check(
            message="Hello, anyone there?",
            recent_context="<user> Hi everyone",
            mentioned=False,
            is_question=True,
            trace_ctx=ctx,
        )

        assert len(ctx.steps) == 1
        step = ctx.steps[0]
        assert step.stage == "engagement"
        assert step.model is not None
        assert step.prompt is not None
        assert step.raw_response is not None
        assert "probability" in step.outputs
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_engagement_tracing.py -v`
Expected: FAIL (trace_ctx parameter doesn't exist)

**Step 3: Modify EngagementChecker.check**

Update `src/bicker_bot/core/engagement.py`:

```python
# Add import at top (after existing imports)
from bicker_bot.tracing import TraceContext

# Modify check method signature
async def check(
    self,
    message: str,
    recent_context: str,
    mentioned: bool = False,
    is_question: bool = False,
    trace_ctx: TraceContext | None = None,  # Add this
) -> EngagementResult:
```

Add tracing after getting the response (before returning):
```python
# After parsing probability, before return
if trace_ctx is not None:
    trace_ctx.add_llm_step(
        stage="engagement",
        inputs={
            "message": message[:200],
            "mentioned": mentioned,
            "is_question": is_question,
        },
        outputs={
            "probability": probability,
        },
        decision=f"P={probability:.0%}",
        model=self._model,
        prompt=user_prompt,
        raw_response=raw,
        thinking=None,
        thought_signatures=None,  # TODO: Extract if available
        token_usage={
            "input": usage.prompt_token_count if usage else 0,
            "output": usage.candidates_token_count if usage else 0,
        } if usage else None,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_engagement_tracing.py -v`
Expected: PASS (or skip if no API keys)

**Step 5: Commit**

```bash
jj commit -m "Add tracing support to EngagementChecker"
```

---

## Task 8: Instrument ContextBuilder with Tracing

**Files:**
- Modify: `src/bicker_bot/core/context.py`

**Step 1: Modify ContextBuilder.build signature**

Add `trace_ctx: TraceContext | None = None` parameter and add tracing.

The context builder is more complex as it has multiple tool rounds. Add a trace step at the end summarizing:
- All tool calls made
- RAG queries and results
- Final summary produced

```python
# Add at end of build() method, before return result:
if trace_ctx is not None:
    trace_ctx.add_llm_step(
        stage="context",
        inputs={
            "message": message[:200],
            "sender": sender,
            "high_intensity_count": len(high_intensity_memories),
        },
        outputs={
            "summary": result.summary,
            "rounds": result.rounds,
            "memories_found": len(result.memories_found),
        },
        decision=f"{result.rounds} rounds, {len(result.memories_found)} memories",
        model=self._model,
        prompt=initial_prompt,
        raw_response=str(result.summary),
        thinking=None,
        thought_signatures=None,
        token_usage=None,  # Aggregate across rounds not easily available
        details={
            "search_queries": result.search_queries,
        },
    )
```

**Step 2: Commit**

```bash
jj commit -m "Add tracing support to ContextBuilder"
```

---

## Task 9: Instrument ResponseGenerator with Tracing

**Files:**
- Modify: `src/bicker_bot/core/responder.py`

**Step 1: Modify ResponseGenerator.generate and _generate_response**

Add `trace_ctx` parameter and capture:
- Full prompt
- Thinking blocks (for Claude)
- Raw response
- Token usage

```python
# In _generate_response, after getting final response:
if trace_ctx is not None:
    trace_ctx.add_llm_step(
        stage="responder",
        inputs={
            "bot": bot.value,
            "message_preview": user_prompt[:200],
        },
        outputs={
            "messages": messages_out,
            "truncated": False,
        },
        decision=f"{len(messages_out)} messages",
        model=self._opus_model,
        prompt=user_prompt,
        raw_response=raw_content or "",
        thinking=thinking_text,  # Extract from response if available
        thought_signatures=None,
        token_usage={
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        },
    )
```

**Step 2: Commit**

```bash
jj commit -m "Add tracing support to ResponseGenerator"
```

---

## Task 10: Wire Tracing Through Orchestrator

**Files:**
- Modify: `src/bicker_bot/orchestrator.py`

**Step 1: Create TraceContext in _process_message and _process_direct_addressed**

At the start of each processing method:
```python
from bicker_bot.tracing import TraceContext, TraceStore

# In __init__, add:
self._trace_store = TraceStore(Path("data/traces.db"))

# At start of _process_message:
ctx = TraceContext(
    channel=channel,
    trigger_messages=[message.content],
    config_snapshot={},  # TODO: Use config loader snapshot
)

# Pass ctx through each component:
gate_result = self._gate.should_respond(..., trace_ctx=ctx)
engagement_result = await self._engagement.check(..., trace_ctx=ctx)
# etc.

# At end, save trace:
ctx.final_result = result.messages
self._trace_store.save(ctx)
```

**Step 2: Commit**

```bash
jj commit -m "Wire TraceContext through orchestrator pipeline"
```

---

## Task 11: Basic Debug Server Structure

**Files:**
- Create: `src/bicker_bot/debug/server.py`
- Create: `src/bicker_bot/debug/templates/base.html`
- Create: `src/bicker_bot/debug/templates/traces.html`
- Create: `tests/test_debug_server.py`

**Step 1: Write basic server test**

Create `tests/test_debug_server.py`:
```python
"""Tests for debug server."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bicker_bot.debug.server import create_app
from bicker_bot.tracing import TraceContext, TraceStore


class TestDebugServer:
    """Tests for debug server routes."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> TraceStore:
        """Create a test trace store."""
        return TraceStore(tmp_path / "traces.db")

    @pytest.fixture
    def client(self, store: TraceStore, tmp_path: Path) -> TestClient:
        """Create a test client."""
        app = create_app(
            trace_store=store,
            config_dir=tmp_path / "config",
        )
        return TestClient(app)

    def test_root_redirects_to_traces(self, client: TestClient):
        """Test that root redirects to traces."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert "/traces" in response.headers["location"]

    def test_traces_list_empty(self, client: TestClient):
        """Test traces list with no traces."""
        response = client.get("/traces")
        assert response.status_code == 200
        assert "No traces" in response.text or "traces" in response.text.lower()

    def test_traces_list_with_data(self, client: TestClient, store: TraceStore):
        """Test traces list with some traces."""
        ctx = TraceContext(
            channel="#test",
            trigger_messages=["hello world"],
            config_snapshot={},
        )
        ctx.final_result = ["hi there"]
        store.save(ctx)

        response = client.get("/traces")
        assert response.status_code == 200
        assert "#test" in response.text
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_debug_server.py -v`
Expected: FAIL (server module doesn't exist)

**Step 3: Implement basic server**

Create `src/bicker_bot/debug/server.py`:
```python
"""FastAPI debug server."""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from bicker_bot.debug.config_loader import ConfigLoader
from bicker_bot.tracing import TraceStore

logger = logging.getLogger(__name__)

# Template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    trace_store: TraceStore,
    config_dir: Path | None = None,
    memory_store=None,  # Optional, for memory browser
    replay_fn=None,  # Optional, for replay functionality
) -> FastAPI:
    """Create the debug server FastAPI app."""
    app = FastAPI(title="Bicker-Bot Debug")

    # Ensure template directory exists
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    # Optional static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Config loader (optional)
    config_loader = None
    if config_dir and config_dir.exists():
        config_loader = ConfigLoader(config_dir)

    @app.get("/", response_class=RedirectResponse)
    async def root():
        """Redirect to traces list."""
        return RedirectResponse(url="/traces", status_code=307)

    @app.get("/traces", response_class=HTMLResponse)
    async def traces_list(request: Request, channel: str = None, bot: str = None):
        """List recent traces."""
        traces = trace_store.recent(limit=50, channel=channel, bot=bot)
        return templates.TemplateResponse(
            "traces.html",
            {
                "request": request,
                "traces": traces,
                "channel_filter": channel,
                "bot_filter": bot,
            },
        )

    @app.get("/traces/{trace_id}", response_class=HTMLResponse)
    async def trace_detail(request: Request, trace_id: str):
        """View a single trace."""
        trace = trace_store.get(trace_id)
        if trace is None:
            return HTMLResponse("Trace not found", status_code=404)
        return templates.TemplateResponse(
            "trace_detail.html",
            {"request": request, "trace": trace},
        )

    return app
```

**Step 4: Create base template**

Create `src/bicker_bot/debug/templates/base.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Bicker-Bot Debug{% endblock %}</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <style>
        :root {
            --bg: #1a1a2e;
            --surface: #16213e;
            --primary: #0f3460;
            --accent: #e94560;
            --text: #eaeaea;
            --text-dim: #888;
            --success: #4ade80;
            --warning: #fbbf24;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'SF Mono', 'Consolas', monospace;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 1rem; }
        header {
            background: var(--surface);
            padding: 1rem;
            border-bottom: 2px solid var(--accent);
        }
        header h1 { font-size: 1.5rem; }
        nav { margin-top: 0.5rem; }
        nav a {
            color: var(--accent);
            text-decoration: none;
            margin-right: 1rem;
        }
        nav a:hover { text-decoration: underline; }
        main { padding: 1rem 0; }
        .card {
            background: var(--surface);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 0.5rem; text-align: left; border-bottom: 1px solid var(--primary); }
        th { color: var(--text-dim); font-weight: normal; }
        .badge {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
        }
        .badge-success { background: var(--success); color: #000; }
        .badge-warning { background: var(--warning); color: #000; }
        .badge-dim { background: var(--primary); }
        a { color: var(--accent); }
        pre {
            background: var(--bg);
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 0.85rem;
        }
        .collapsible { cursor: pointer; }
        .collapsible::before { content: '▶ '; }
        .collapsible.open::before { content: '▼ '; }
        .collapsible-content { display: none; margin-top: 0.5rem; }
        .collapsible.open + .collapsible-content { display: block; }
        button {
            background: var(--accent);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover { opacity: 0.9; }
        .text-dim { color: var(--text-dim); }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>Bicker-Bot Debug</h1>
            <nav>
                <a href="/traces">Traces</a>
                <a href="/memories">Memories</a>
                <a href="#" id="config-btn">Config</a>
            </nav>
        </div>
    </header>
    <main class="container">
        {% block content %}{% endblock %}
    </main>
</body>
</html>
```

**Step 5: Create traces list template**

Create `src/bicker_bot/debug/templates/traces.html`:
```html
{% extends "base.html" %}

{% block title %}Traces - Bicker-Bot Debug{% endblock %}

{% block content %}
<div class="card">
    <h2>Recent Traces</h2>

    {% if traces %}
    <table>
        <thead>
            <tr>
                <th>Time</th>
                <th>Channel</th>
                <th>Bot</th>
                <th>Trigger</th>
                <th>Outcome</th>
            </tr>
        </thead>
        <tbody>
            {% for trace in traces %}
            <tr>
                <td><a href="/traces/{{ trace.id }}">{{ trace.created_at.strftime('%H:%M:%S') }}</a></td>
                <td>{{ trace.channel }}</td>
                <td>{{ trace.bot or '-' }}</td>
                <td>{{ trace.trigger_text[:50] }}{% if trace.trigger_text|length > 50 %}...{% endif %}</td>
                <td>
                    {% if trace.outcome == 'responded' %}
                    <span class="badge badge-success">responded</span>
                    {% elif trace.outcome == 'declined_gate' %}
                    <span class="badge badge-dim">gate</span>
                    {% else %}
                    <span class="badge badge-warning">engagement</span>
                    {% endif %}
                    {% if trace.is_replay %}<span class="badge badge-dim">replay</span>{% endif %}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% else %}
    <p class="text-dim">No traces recorded yet.</p>
    {% endif %}
</div>
{% endblock %}
```

**Step 6: Create trace detail template**

Create `src/bicker_bot/debug/templates/trace_detail.html`:
```html
{% extends "base.html" %}

{% block title %}Trace {{ trace.id[:8] }} - Bicker-Bot Debug{% endblock %}

{% block content %}
<div class="card">
    <h2>Trace {{ trace.id[:8] }}...</h2>
    <p class="text-dim">{{ trace.started_at }} | {{ trace.channel }}</p>

    <h3 style="margin-top: 1rem;">Trigger</h3>
    <pre>{{ trace.trigger_messages | join('\n') }}</pre>

    {% if trace.final_result %}
    <h3 style="margin-top: 1rem;">Response</h3>
    <pre>{{ trace.final_result | join('\n') }}</pre>
    {% endif %}
</div>

<div class="card">
    <h2>Pipeline Steps</h2>

    {% for step in trace.steps %}
    <div style="margin-bottom: 1rem; border-left: 3px solid var(--accent); padding-left: 1rem;">
        <h3 class="collapsible" onclick="this.classList.toggle('open')">
            {{ step.stage }} - {{ step.decision }}
        </h3>
        <div class="collapsible-content">
            <h4>Inputs</h4>
            <pre>{{ step.inputs | tojson(indent=2) }}</pre>

            <h4>Outputs</h4>
            <pre>{{ step.outputs | tojson(indent=2) }}</pre>

            {% if step.prompt %}
            <h4>Prompt</h4>
            <pre>{{ step.prompt }}</pre>
            {% endif %}

            {% if step.raw_response %}
            <h4>Raw Response</h4>
            <pre>{{ step.raw_response }}</pre>
            {% endif %}

            {% if step.thinking %}
            <h4>Thinking</h4>
            <pre>{{ step.thinking }}</pre>
            {% endif %}

            {% if step.token_usage %}
            <p class="text-dim">Tokens: {{ step.token_usage.input }} in / {{ step.token_usage.output }} out</p>
            {% endif %}
        </div>
    </div>
    {% endfor %}
</div>

<div class="card">
    <button hx-get="/traces/{{ trace.id }}/replay" hx-target="#replay-result">
        Replay with Current Config
    </button>
    <div id="replay-result"></div>
</div>
{% endblock %}
```

**Step 7: Run test to verify it passes**

Run: `uv run pytest tests/test_debug_server.py -v`
Expected: PASS

**Step 8: Commit**

```bash
jj commit -m "Add basic debug server with traces list and detail views"
```

---

## Task 12: Memory Browser UI

**Files:**
- Modify: `src/bicker_bot/debug/server.py`
- Create: `src/bicker_bot/debug/templates/memories.html`
- Create: `src/bicker_bot/debug/templates/partials/memory_results.html`

**Step 1: Add memory routes to server**

Add to `server.py`:
```python
@app.get("/memories", response_class=HTMLResponse)
async def memories_list(request: Request):
    """Memory browser."""
    if memory_store is None:
        return HTMLResponse("Memory store not configured", status_code=503)
    return templates.TemplateResponse(
        "memories.html",
        {"request": request, "collections": ["memories"]},
    )

@app.get("/memories/search", response_class=HTMLResponse)
async def memories_search(
    request: Request,
    query: str = "",
    collection: str = "memories",
    limit: int = 20,
):
    """Search memories (htmx partial)."""
    if memory_store is None:
        return HTMLResponse("Memory store not configured")

    if query:
        results = memory_store.search(query=query, limit=limit)
    else:
        # Browse mode - get recent
        results = []  # TODO: Add browse method to memory store

    return templates.TemplateResponse(
        "partials/memory_results.html",
        {"request": request, "results": results, "query": query},
    )

@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory."""
    if memory_store is None:
        return {"error": "Memory store not configured"}
    success = memory_store.delete(memory_id)
    return {"success": success}
```

**Step 2: Create memories template and partials**

(Similar structure to traces, with search box and results table)

**Step 3: Commit**

```bash
jj commit -m "Add memory browser UI with search and delete"
```

---

## Task 13: Config Modal and Session Overrides

**Files:**
- Modify: `src/bicker_bot/debug/server.py`
- Create: `src/bicker_bot/debug/templates/partials/config_modal.html`
- Add session handling

**Step 1: Add config routes**

```python
@app.get("/config/modal", response_class=HTMLResponse)
async def config_modal(request: Request):
    """Config editor modal (htmx)."""
    prompts = config_loader.list_prompts() if config_loader else []
    policies = config_loader.get_policies() if config_loader else {}
    # Get overrides from session/localStorage (via header)
    overrides = {}  # TODO: Parse from request
    return templates.TemplateResponse(
        "partials/config_modal.html",
        {
            "request": request,
            "prompts": prompts,
            "policies": policies,
            "overrides": overrides,
        },
    )

@app.post("/config/reload")
async def reload_config():
    """Hot-reload config from disk."""
    if config_loader:
        config_loader.reload()
        return {"success": True}
    return {"success": False, "error": "No config loader"}
```

**Step 2: Commit**

```bash
jj commit -m "Add config modal with session overrides support"
```

---

## Task 14: Replay Mechanism

**Files:**
- Modify: `src/bicker_bot/debug/server.py`
- Modify: `src/bicker_bot/orchestrator.py`
- Create: `src/bicker_bot/debug/templates/partials/replay_comparison.html`

**Step 1: Add replay route**

```python
@app.post("/traces/{trace_id}/replay", response_class=HTMLResponse)
async def replay_trace(request: Request, trace_id: str):
    """Replay a trace with current/modified config."""
    if replay_fn is None:
        return HTMLResponse("Replay not available", status_code=503)

    original = trace_store.get(trace_id)
    if original is None:
        return HTMLResponse("Trace not found", status_code=404)

    # Get config overrides from request body
    overrides = {}  # TODO: Parse from request

    # Run replay
    replay_result = await replay_fn(trace_id, overrides)

    return templates.TemplateResponse(
        "partials/replay_comparison.html",
        {
            "request": request,
            "original": original,
            "replayed": replay_result.replayed,
            "config_diff": replay_result.config_diff,
            "decision_diffs": replay_result.decision_diffs,
        },
    )
```

**Step 2: Add replay method to Orchestrator**

Add `async def replay(self, trace_id: str, config_overrides: dict | None = None) -> ReplayResult` that:
1. Loads original trace
2. Builds effective config
3. Runs pipeline with `dry_run=True`
4. Returns comparison

**Step 3: Commit**

```bash
jj commit -m "Add replay mechanism with A/B comparison"
```

---

## Task 15: Integrate Debug Server into Orchestrator Startup

**Files:**
- Modify: `src/bicker_bot/orchestrator.py`

**Step 1: Start debug server as background task**

```python
import asyncio
import uvicorn
from bicker_bot.debug.server import create_app

async def start(self) -> None:
    """Start the IRC connection and debug server."""
    # Create debug server
    app = create_app(
        trace_store=self._trace_store,
        config_dir=Path("config"),
        memory_store=self._memory_store,
        replay_fn=self.replay,
    )

    # Start debug server in background
    config = uvicorn.Config(app, host="127.0.0.1", port=8080, log_level="warning")
    server = uvicorn.Server(config)
    asyncio.create_task(server.serve())

    # Continue with IRC connection
    self._irc = IRCClient(...)
    await self._irc.connect()
    await self._irc.run_forever()
```

**Step 2: Commit**

```bash
jj commit -m "Start debug server alongside IRC client"
```

---

## Task 16: Create Default Config Files

**Files:**
- Create: `config/prompts/merry.md`
- Create: `config/prompts/hachiman.md`
- Create: `config/prompts/engagement.md`
- Create: `config/prompts/context.md`
- Create: `config/policies.yaml`

**Step 1: Migrate existing prompts to files**

Extract prompts from:
- `src/bicker_bot/personalities/merry.py`
- `src/bicker_bot/personalities/hachiman.py`
- `src/bicker_bot/core/engagement.py` (ENGAGEMENT_SYSTEM_PROMPT)
- `src/bicker_bot/core/context.py` (get_context_system_prompt)

Format as markdown with frontmatter.

**Step 2: Commit**

```bash
jj commit -m "Add config files for prompts and policies"
```

---

## Task 17: Final Integration Testing

**Files:**
- Create: `tests/test_debug_integration.py`

**Step 1: Write integration test**

Test the full flow:
1. Create orchestrator with debug server
2. Process a message through pipeline
3. Verify trace was stored
4. Query debug server API
5. Verify trace appears in list
6. Verify trace detail loads

**Step 2: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

**Step 3: Commit**

```bash
jj commit -m "Add integration tests for debug observability system"
```

---

## Summary

| Task | Component | Commits |
|------|-----------|---------|
| 1 | Dependencies | 1 |
| 2-3 | TraceStep, TraceContext | 2 |
| 4 | TraceStore | 1 |
| 5 | ConfigLoader | 1 |
| 6-9 | Instrument pipeline | 4 |
| 10 | Wire orchestrator | 1 |
| 11 | Basic server | 1 |
| 12 | Memory browser | 1 |
| 13 | Config modal | 1 |
| 14 | Replay mechanism | 1 |
| 15 | Server integration | 1 |
| 16 | Default config files | 1 |
| 17 | Integration tests | 1 |

**Total: 17 tasks, ~17 commits**
