"""Tests for tracing data model."""

from bicker_bot.tracing.context import TraceContext, TraceStep


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
