"""Tests for engagement checker tracing."""

import os
import pytest

from bicker_bot.tracing import TraceContext


class TestEngagementTracing:
    """Tests for engagement tracing integration."""

    @pytest.mark.asyncio
    async def test_engagement_adds_llm_trace_step(self, has_api_keys: bool):
        """Test that engagement checker adds LLM step to trace."""
        if not has_api_keys:
            pytest.skip("API keys not available")

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

    @pytest.mark.asyncio
    async def test_engagement_without_trace_ctx(self, has_api_keys: bool):
        """Test that engagement checker works without trace context (backward compatibility)."""
        if not has_api_keys:
            pytest.skip("API keys not available")

        from bicker_bot.core.engagement import EngagementChecker

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        checker = EngagementChecker(api_key)

        # Should work without trace_ctx parameter
        result = await checker.check(
            message="Hello, anyone there?",
            recent_context="<user> Hi everyone",
            mentioned=False,
            is_question=True,
        )

        assert result is not None
        assert 0.0 <= result.probability <= 1.0
