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

    def test_trace_detail_found(self, client: TestClient, store: TraceStore):
        """Test viewing a trace detail page."""
        ctx = TraceContext(
            channel="#test",
            trigger_messages=["hello world"],
            config_snapshot={},
        )
        ctx.add_step(
            stage="gate",
            inputs={"messages": ["hello world"]},
            outputs={"probability": 0.5},
            decision="Passed: P=0.5",
        )
        ctx.final_result = ["hi there"]
        store.save(ctx)

        response = client.get(f"/traces/{ctx.id}")
        assert response.status_code == 200
        assert "#test" in response.text
        assert "hello world" in response.text

    def test_trace_detail_not_found(self, client: TestClient):
        """Test viewing a nonexistent trace."""
        response = client.get("/traces/nonexistent-id")
        assert response.status_code == 404
        assert "not found" in response.text.lower()

    def test_traces_filter_by_channel(self, client: TestClient, store: TraceStore):
        """Test filtering traces by channel."""
        for channel in ["#alpha", "#alpha", "#beta"]:
            ctx = TraceContext(
                channel=channel,
                trigger_messages=["test"],
                config_snapshot={},
            )
            ctx.final_result = ["response"]
            store.save(ctx)

        response = client.get("/traces?channel=%23alpha")
        assert response.status_code == 200
        assert "#alpha" in response.text

    def test_traces_filter_by_bot(self, client: TestClient, store: TraceStore):
        """Test filtering traces by bot."""
        ctx = TraceContext(
            channel="#test",
            trigger_messages=["test"],
            config_snapshot={},
        )
        ctx.add_step(
            stage="selector",
            inputs={},
            outputs={"selected": "hachiman"},
            decision="Selected hachiman",
        )
        ctx.final_result = ["response"]
        store.save(ctx)

        response = client.get("/traces?bot=hachiman")
        assert response.status_code == 200


class TestConfigModal:
    """Tests for config modal routes."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> TraceStore:
        """Create a test trace store."""
        return TraceStore(tmp_path / "traces.db")

    @pytest.fixture
    def config_dir(self, tmp_path: Path) -> Path:
        """Create a config directory with test files."""
        config_path = tmp_path / "config"
        prompts_dir = config_path / "prompts"
        prompts_dir.mkdir(parents=True)

        # Create test prompt files
        (prompts_dir / "merry.md").write_text("""---
model: gemini-3-pro-preview
max_tokens: 1024
temperature: 0.9
---

You are Merry Nightmare, a dream demon.
""")
        (prompts_dir / "hachiman.md").write_text("""---
model: claude-opus-4-5
max_tokens: 2048
---

You are Hachiman Hikigaya.
""")

        # Create policies file
        (config_path / "policies.yaml").write_text("""
gate:
  base_prob: 0.05
  decay_factor: 0.5
""")

        return config_path

    @pytest.fixture
    def client(self, store: TraceStore, config_dir: Path) -> TestClient:
        """Create a test client with config."""
        app = create_app(
            trace_store=store,
            config_dir=config_dir,
        )
        return TestClient(app)

    @pytest.fixture
    def client_no_config(self, store: TraceStore, tmp_path: Path) -> TestClient:
        """Create a test client without config."""
        app = create_app(
            trace_store=store,
            config_dir=tmp_path / "nonexistent",
        )
        return TestClient(app)

    def test_config_modal_with_prompts(self, client: TestClient):
        """Test config modal shows prompts."""
        response = client.get("/config/modal")
        assert response.status_code == 200
        assert "merry.md" in response.text
        assert "hachiman.md" in response.text
        assert "policies.yaml" in response.text

    def test_config_modal_shows_prompt_content(self, client: TestClient):
        """Test config modal displays prompt content."""
        response = client.get("/config/modal")
        assert response.status_code == 200
        assert "Merry Nightmare" in response.text
        assert "dream demon" in response.text

    def test_config_modal_shows_prompt_metadata(self, client: TestClient):
        """Test config modal displays prompt metadata."""
        response = client.get("/config/modal")
        assert response.status_code == 200
        assert "gemini-3-pro-preview" in response.text
        assert "1024" in response.text

    def test_config_modal_shows_policies(self, client: TestClient):
        """Test config modal displays policies."""
        response = client.get("/config/modal")
        assert response.status_code == 200
        assert "base_prob" in response.text
        assert "0.05" in response.text

    def test_config_modal_no_config(self, client_no_config: TestClient):
        """Test config modal works without config."""
        response = client_no_config.get("/config/modal")
        assert response.status_code == 200
        # Should still render but with empty content
        assert "policies.yaml" in response.text

    def test_config_reload(self, client: TestClient, config_dir: Path):
        """Test config reload endpoint."""
        response = client.post("/config/reload")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_config_reload_no_config(self, client_no_config: TestClient):
        """Test config reload without config loader."""
        response = client_no_config.post("/config/reload")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data
