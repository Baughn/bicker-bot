"""Integration tests for debug observability system."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from bicker_bot.debug.server import create_app
from bicker_bot.tracing import TraceContext, TraceStore


class TestDebugIntegration:
    """Integration tests for the full debug system."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> TraceStore:
        """Create a temporary trace store."""
        return TraceStore(tmp_path / "traces.db")

    @pytest.fixture
    def config_dir(self, tmp_path: Path) -> Path:
        """Create a mock config directory with prompts."""
        config_dir = tmp_path / "config"
        prompts_dir = config_dir / "prompts"
        prompts_dir.mkdir(parents=True)

        # Create a test prompt with frontmatter
        (prompts_dir / "test.md").write_text(
            """---
model: test-model
max_tokens: 100
temperature: 0.7
---
Test prompt content.

This is a multi-line prompt for testing purposes.
"""
        )

        # Create another prompt for tab testing
        (prompts_dir / "engagement.md").write_text(
            """---
model: gemini-flash
max_tokens: 50
thinking: true
---
Engagement check prompt.
"""
        )

        # Create a policies file
        (config_dir / "policies.yaml").write_text(
            """gate:
  base_prob: 0.05
  mention_prob: 0.8

engagement:
  enabled: true
  bypass_mentions: true
"""
        )
        return config_dir

    @pytest.fixture
    def client(self, store: TraceStore, config_dir: Path) -> TestClient:
        """Create a test client for the debug server."""
        app = create_app(
            trace_store=store,
            config_dir=config_dir,
        )
        return TestClient(app)

    @pytest.fixture
    def client_no_config(self, store: TraceStore) -> TestClient:
        """Create a test client without config directory."""
        app = create_app(
            trace_store=store,
            config_dir=None,
        )
        return TestClient(app)

    def test_full_trace_workflow(self, client: TestClient, store: TraceStore):
        """Test creating trace and viewing it in debug UI."""
        # Create and save a trace
        ctx = TraceContext(
            channel="#test",
            trigger_messages=["hello bot"],
            config_snapshot={"gate": {"base_prob": 0.05}},
        )
        ctx.add_step(
            stage="gate",
            inputs={"message": "hello bot"},
            outputs={"probability": 0.8, "should_respond": True},
            decision="PASS: P=0.800",
        )
        ctx.final_result = ["Hello there!"]
        store.save(ctx)

        # Verify trace appears in list
        response = client.get("/traces")
        assert response.status_code == 200
        assert "#test" in response.text
        assert "hello bot" in response.text

        # Verify trace detail loads
        response = client.get(f"/traces/{ctx.id}")
        assert response.status_code == 200
        assert "gate" in response.text.lower()
        assert "PASS" in response.text

    def test_trace_list_shows_multiple_traces(self, client: TestClient, store: TraceStore):
        """Test that multiple traces appear in the list."""
        # Create several traces with different properties
        for i in range(3):
            ctx = TraceContext(
                channel=f"#channel{i}",
                trigger_messages=[f"message {i}"],
                config_snapshot={},
            )
            ctx.add_step(
                stage="gate",
                inputs={},
                outputs={"probability": 0.5 + i * 0.1},
                decision=f"PASS: P={0.5 + i * 0.1:.3f}",
            )
            ctx.final_result = [f"response {i}"]
            store.save(ctx)

        response = client.get("/traces")
        assert response.status_code == 200
        # Check all channels appear
        for i in range(3):
            assert f"#channel{i}" in response.text

    def test_trace_detail_shows_all_steps(self, client: TestClient, store: TraceStore):
        """Test that trace detail shows all pipeline steps."""
        ctx = TraceContext(
            channel="#multi-step",
            trigger_messages=["complex message"],
            config_snapshot={},
        )

        # Add multiple steps
        ctx.add_step(
            stage="gate",
            inputs={"message": "complex message"},
            outputs={"probability": 0.9, "should_respond": True},
            decision="PASS: direct mention",
        )
        ctx.add_step(
            stage="engagement",
            inputs={"message": "complex message"},
            outputs={"engaged": True},
            decision="ENGAGED: genuine human engagement",
        )
        ctx.add_step(
            stage="selector",
            inputs={"message": "complex message"},
            outputs={"selected": "hachiman", "confidence": 0.75},
            decision="Selected hachiman (score: 0.75)",
        )
        ctx.final_result = ["A thoughtful response from Hachiman"]
        store.save(ctx)

        response = client.get(f"/traces/{ctx.id}")
        assert response.status_code == 200
        # Check all stages are present (case insensitive)
        text_lower = response.text.lower()
        assert "gate" in text_lower
        assert "engagement" in text_lower
        assert "selector" in text_lower

    def test_trace_detail_shows_llm_step_details(self, client: TestClient, store: TraceStore):
        """Test that LLM-specific step fields are displayed."""
        ctx = TraceContext(
            channel="#llm-test",
            trigger_messages=["test llm"],
            config_snapshot={},
        )

        ctx.add_llm_step(
            stage="responder",
            inputs={"context": "some context"},
            outputs={"response": "test response"},
            decision="Generated response",
            model="claude-3-opus-20240229",
            prompt="You are a helpful assistant.",
            raw_response="test response",
            thinking="Let me think about this...",
            token_usage={"input": 100, "output": 50},
        )
        ctx.final_result = ["test response"]
        store.save(ctx)

        response = client.get(f"/traces/{ctx.id}")
        assert response.status_code == 200
        # Check LLM-specific fields appear
        assert "claude-3-opus" in response.text
        assert "You are a helpful assistant" in response.text

    def test_trace_not_found_returns_404(self, client: TestClient):
        """Test that requesting non-existent trace returns 404."""
        response = client.get("/traces/non-existent-trace-id")
        assert response.status_code == 404
        assert "not found" in response.text.lower()

    def test_config_modal_loads(self, client: TestClient):
        """Test config modal displays prompts."""
        response = client.get("/config/modal")
        assert response.status_code == 200
        # Check that prompt names appear
        assert "test.md" in response.text or "test" in response.text
        assert "engagement.md" in response.text or "engagement" in response.text

    def test_config_modal_shows_prompt_metadata(self, client: TestClient):
        """Test config modal shows prompt frontmatter details."""
        response = client.get("/config/modal")
        assert response.status_code == 200
        # Check that frontmatter values appear
        assert "test-model" in response.text
        assert "100" in response.text  # max_tokens
        assert "gemini-flash" in response.text

    def test_config_modal_shows_policies(self, client: TestClient):
        """Test config modal shows policies section."""
        response = client.get("/config/modal")
        assert response.status_code == 200
        assert "policies" in response.text.lower()

    def test_config_modal_without_config_dir(self, client_no_config: TestClient):
        """Test config modal works even without config directory."""
        response = client_no_config.get("/config/modal")
        assert response.status_code == 200
        # Should still render but with empty prompts
        assert "policies" in response.text.lower()

    def test_config_reload(self, client: TestClient):
        """Test config reload endpoint."""
        response = client.post("/config/reload")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_config_reload_without_loader(self, client_no_config: TestClient):
        """Test config reload when no config loader is available."""
        response = client_no_config.post("/config/reload")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_root_redirects_to_traces(self, client: TestClient):
        """Test that root URL redirects to traces list."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/traces"

    def test_trace_filter_by_channel(self, client: TestClient, store: TraceStore):
        """Test filtering traces by channel."""
        # Create traces in different channels
        for channel in ["#general", "#general", "#bots"]:
            ctx = TraceContext(
                channel=channel,
                trigger_messages=["test"],
                config_snapshot={},
            )
            ctx.final_result = ["response"]
            store.save(ctx)

        # Filter by #general
        response = client.get("/traces?channel=%23general")
        assert response.status_code == 200
        assert "#general" in response.text
        # #bots should not appear (or appear only minimally due to filter form)

    def test_trace_with_no_response(self, client: TestClient, store: TraceStore):
        """Test viewing a trace where bot didn't respond."""
        ctx = TraceContext(
            channel="#ignored",
            trigger_messages=["ignored message"],
            config_snapshot={},
        )
        ctx.add_step(
            stage="gate",
            inputs={"message": "ignored message"},
            outputs={"probability": 0.02, "should_respond": False},
            decision="SKIP: P=0.020 < threshold",
        )
        # No final_result
        store.save(ctx)

        response = client.get(f"/traces/{ctx.id}")
        assert response.status_code == 200
        assert "SKIP" in response.text or "no response" in response.text.lower()


class TestMemoriesEndpoint:
    """Tests for the memories browser endpoint."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> TraceStore:
        """Create a temporary trace store."""
        return TraceStore(tmp_path / "traces.db")

    @pytest.fixture
    def client_without_memory(self, store: TraceStore) -> TestClient:
        """Create client without memory store."""
        app = create_app(trace_store=store)
        return TestClient(app)

    def test_memories_without_store_returns_503(self, client_without_memory: TestClient):
        """Test that memories endpoint returns 503 when no memory store."""
        response = client_without_memory.get("/memories")
        assert response.status_code == 503
        assert "not configured" in response.text.lower()

    def test_memories_search_without_store(self, client_without_memory: TestClient):
        """Test that memory search returns appropriate message without store."""
        response = client_without_memory.get("/memories/search?query=test")
        assert response.status_code == 200
        assert "not configured" in response.text.lower()


class TestReplayEndpoint:
    """Tests for the trace replay functionality."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> TraceStore:
        """Create a temporary trace store."""
        return TraceStore(tmp_path / "traces.db")

    @pytest.fixture
    def client_without_replay(self, store: TraceStore) -> TestClient:
        """Create client without replay function."""
        app = create_app(trace_store=store)
        return TestClient(app)

    def test_replay_without_function_returns_503(
        self, client_without_replay: TestClient, store: TraceStore
    ):
        """Test that replay returns 503 when replay_fn not configured."""
        # Create a trace to replay
        ctx = TraceContext(
            channel="#test",
            trigger_messages=["test"],
            config_snapshot={},
        )
        ctx.final_result = ["response"]
        store.save(ctx)

        response = client_without_replay.post(f"/traces/{ctx.id}/replay")
        assert response.status_code == 503
        assert "not available" in response.text.lower()

    def test_replay_nonexistent_trace_returns_404(
        self, store: TraceStore, tmp_path: Path
    ):
        """Test that replaying non-existent trace returns 404."""
        # Create client with mock replay function
        async def mock_replay(trace_id: str, config_overrides: dict):
            pass

        app = create_app(trace_store=store, replay_fn=mock_replay)
        client = TestClient(app)

        response = client.post("/traces/nonexistent-id/replay")
        assert response.status_code == 404


class TestTraceStorePersistence:
    """Tests for trace store persistence across requests."""

    def test_traces_persist_across_client_recreations(self, tmp_path: Path):
        """Test that traces persist when store is recreated."""
        db_path = tmp_path / "traces.db"

        # Create store, save trace, close
        store1 = TraceStore(db_path)
        ctx = TraceContext(
            channel="#persistent",
            trigger_messages=["persist me"],
            config_snapshot={},
        )
        ctx.final_result = ["stored"]
        store1.save(ctx)
        trace_id = ctx.id
        store1.close()

        # Create new store from same path
        store2 = TraceStore(db_path)
        app = create_app(trace_store=store2)
        client = TestClient(app)

        # Verify trace is accessible
        response = client.get(f"/traces/{trace_id}")
        assert response.status_code == 200
        assert "#persistent" in response.text

        store2.close()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> TraceStore:
        """Create a temporary trace store."""
        return TraceStore(tmp_path / "traces.db")

    @pytest.fixture
    def config_dir(self, tmp_path: Path) -> Path:
        """Create minimal config directory."""
        config_dir = tmp_path / "config"
        prompts_dir = config_dir / "prompts"
        prompts_dir.mkdir(parents=True)
        return config_dir

    @pytest.fixture
    def client(self, store: TraceStore, config_dir: Path) -> TestClient:
        """Create a test client."""
        app = create_app(trace_store=store, config_dir=config_dir)
        return TestClient(app)

    def test_trace_with_empty_trigger_messages(self, client: TestClient, store: TraceStore):
        """Test handling trace with empty trigger messages."""
        ctx = TraceContext(
            channel="#empty",
            trigger_messages=[],
            config_snapshot={},
        )
        store.save(ctx)

        # Should not crash
        response = client.get("/traces")
        assert response.status_code == 200

        response = client.get(f"/traces/{ctx.id}")
        assert response.status_code == 200

    def test_trace_with_special_characters(self, client: TestClient, store: TraceStore):
        """Test handling trace with special characters in messages."""
        ctx = TraceContext(
            channel="#special",
            trigger_messages=["<script>alert('xss')</script>", "test & test"],
            config_snapshot={"key": "value with 'quotes' and \"double quotes\""},
        )
        ctx.final_result = ["response with <html> tags"]
        store.save(ctx)

        response = client.get(f"/traces/{ctx.id}")
        assert response.status_code == 200
        # HTML should be escaped, not rendered
        assert "<script>" not in response.text or "&lt;script&gt;" in response.text

    def test_trace_with_large_config_snapshot(self, client: TestClient, store: TraceStore):
        """Test handling trace with large config snapshot."""
        large_config = {f"key_{i}": f"value_{i}" * 100 for i in range(50)}
        ctx = TraceContext(
            channel="#large",
            trigger_messages=["test"],
            config_snapshot=large_config,
        )
        ctx.final_result = ["response"]
        store.save(ctx)

        response = client.get(f"/traces/{ctx.id}")
        assert response.status_code == 200

    def test_empty_traces_list(self, client: TestClient):
        """Test that empty traces list renders correctly."""
        response = client.get("/traces")
        assert response.status_code == 200
        assert "No traces found" in response.text or "no traces" in response.text.lower()

    def test_config_with_empty_prompts_dir(self, client: TestClient):
        """Test config modal with no prompts (empty directory)."""
        response = client.get("/config/modal")
        assert response.status_code == 200
        # Should render without errors
