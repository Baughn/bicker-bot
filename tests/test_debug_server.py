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
