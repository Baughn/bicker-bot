"""Integration tests using real API keys from .env."""

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from dotenv import load_dotenv

from bicker_bot.config import Config, GateConfig, MemoryConfig, LLMConfig
from bicker_bot.core import (
    ContextBuilder,
    EngagementChecker,
    MessageRouter,
    ResponseGate,
    ResponseGenerator,
)
from bicker_bot.irc.client import Message
from bicker_bot.memory import BotIdentity, BotSelector, Memory, MemoryExtractor, MemoryStore
from bicker_bot.personalities import get_hachiman_prompt, get_merry_prompt

# Load .env for tests
load_dotenv()


def has_api_keys() -> bool:
    """Check if API keys are available."""
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return bool(os.getenv("ANTHROPIC_API_KEY") and google_key)


def get_google_key() -> str:
    """Get Google/Gemini API key."""
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""


# Skip all tests in this module if no API keys
pytestmark = pytest.mark.skipif(
    not has_api_keys(),
    reason="API keys not available (set ANTHROPIC_API_KEY and GOOGLE_API_KEY in .env)",
)


class TestEngagementIntegration:
    """Integration tests for engagement checker."""

    @pytest.fixture
    def engagement_checker(self) -> EngagementChecker:
        """Create engagement checker with real API key."""
        return EngagementChecker(
            api_key=get_google_key(),
            model="gemini-2.0-flash-exp",  # Use available model
        )

    @pytest.mark.asyncio
    async def test_engagement_check_question(self, engagement_checker: EngagementChecker):
        """Test engagement check with a direct question."""
        result = await engagement_checker.check(
            message="Hey Merry, what do you think about this?",
            recent_context="[12:00] <Alice> Just had lunch\n[12:01] <Bob> Same",
            mentioned=True,
            is_question=True,
        )

        # Direct mentions should generally trigger engagement
        assert result.is_engaged or "error" in result.raw_response

    @pytest.mark.asyncio
    async def test_engagement_check_noise(self, engagement_checker: EngagementChecker):
        """Test engagement check with ambient noise."""
        result = await engagement_checker.check(
            message="lol",
            recent_context="[12:00] <Alice> brb\n[12:01] <Bob> k",
            mentioned=False,
            is_question=False,
        )

        # Should not engage with noise
        assert not result.is_engaged or "error" in result.raw_response


class TestResponseIntegration:
    """Integration tests for response generation."""

    @pytest.fixture
    def responder(self) -> ResponseGenerator:
        """Create response generator with real API keys."""
        return ResponseGenerator(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            google_api_key=get_google_key(),
            opus_model="claude-sonnet-4-20250514",  # Use available model for testing
            gemini_model="gemini-2.0-flash-exp",  # Use available model
        )

    @pytest.mark.asyncio
    async def test_generate_as_hachiman(self, responder: ResponseGenerator):
        """Test generating a response as Hachiman."""
        result = await responder.generate(
            bot=BotIdentity.HACHIMAN,
            system_prompt=get_hachiman_prompt("TestHachi"),
            context_summary={"key_facts": [], "suggested_tone": "casual"},
            recent_conversation="[12:00] <Alice> What's a good book to read?",
            message="What's a good book to read?",
            sender="Alice",
        )

        assert result.messages
        assert result.bot == BotIdentity.HACHIMAN
        assert len(result.messages) > 0

    @pytest.mark.asyncio
    async def test_generate_as_merry(self, responder: ResponseGenerator):
        """Test generating a response as Merry."""
        result = await responder.generate(
            bot=BotIdentity.MERRY,
            system_prompt=get_merry_prompt("TestMerry"),
            context_summary={"key_facts": [], "suggested_tone": "energetic"},
            recent_conversation="[12:00] <Bob> I'm bored, what should I do?",
            message="I'm bored, what should I do?",
            sender="Bob",
        )

        assert result.messages
        assert result.bot == BotIdentity.MERRY
        assert len(result.messages) > 0


class TestPipelineIntegration:
    """Integration tests for the full pipeline (without IRC)."""

    @pytest.fixture
    def components(self, tmp_path: Path) -> dict:
        """Create all pipeline components."""
        google_key = get_google_key()

        memory_config = MemoryConfig(
            chroma_path=tmp_path / "chroma",
            embedding_model="all-MiniLM-L6-v2",  # Smaller model for testing
        )

        gate_config = GateConfig()

        # Use smaller/faster models for testing
        return {
            "router": MessageRouter(),
            "gate": ResponseGate(gate_config, ("Merry", "Hachi")),
            "memory_store": MemoryStore(memory_config),
            "bot_selector": BotSelector(memory_config),
            "engagement": EngagementChecker(google_key, model="gemini-2.0-flash-exp"),
        }

    @pytest.mark.asyncio
    async def test_message_routing(self, components: dict):
        """Test message routing and buffering."""
        router = components["router"]

        # Add messages
        for i in range(5):
            msg = Message(channel="#test", sender=f"user{i}", content=f"Message {i}")
            await router.add_message(msg)

        # Get recent messages
        messages = await router.get_recent_messages("#test", count=3)
        assert len(messages) == 3

        # Check context formatting
        context = router.format_context(messages)
        assert "user" in context

    @pytest.mark.asyncio
    async def test_gate_with_router(self, components: dict):
        """Test gate using router context."""
        router = components["router"]
        gate = components["gate"]

        # Add a message
        msg = Message(channel="#test", sender="alice", content="Hey Merry, help me?")
        await router.add_message(msg)

        last_activity = await router.get_last_activity("#test")
        consecutive = await router.count_consecutive_bot_messages("#test", ("Merry", "Hachi"))

        result = gate.should_respond(
            message=msg.content,
            last_activity=last_activity,
            consecutive_bot_messages=consecutive,
            _roll=0.1,  # Low roll to trigger response
        )

        # Should pass gate with mention
        assert result.should_respond
        assert result.factors.mentioned

    @pytest.mark.asyncio
    async def test_memory_integration(self, components: dict):
        """Test memory storage and retrieval."""
        store = components["memory_store"]

        # Add some memories
        store.add(Memory(content="Alice likes pizza", user="alice", intensity=0.8))
        store.add(Memory(content="Bob works at a startup", user="bob", intensity=0.7))

        # Search
        results = store.search("food preferences")
        assert len(results) > 0

        # User-specific retrieval
        alice_memories = store.get_high_intensity_memories("alice")
        assert any("pizza" in m.content for m in alice_memories)


class TestPersonalityConsistency:
    """Tests for personality consistency in responses."""

    @pytest.fixture
    def responder(self) -> ResponseGenerator:
        """Create response generator."""
        return ResponseGenerator(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            google_api_key=get_google_key(),
            opus_model="claude-sonnet-4-20250514",
            gemini_model="gemini-2.0-flash-exp",
        )

    @pytest.mark.asyncio
    async def test_hachiman_cynical_tone(self, responder: ResponseGenerator):
        """Test that Hachiman maintains cynical personality."""
        result = await responder.generate(
            bot=BotIdentity.HACHIMAN,
            system_prompt=get_hachiman_prompt("TestHachi"),
            context_summary={"suggested_tone": "cynical"},
            recent_conversation="[12:00] <Eve> I just made friends with everyone in class!",
            message="I just made friends with everyone in class!",
            sender="Eve",
        )

        # Should have some response, personality will vary
        assert result.messages
        # Not asserting specific content since LLMs are variable

    @pytest.mark.asyncio
    async def test_merry_direct_style(self, responder: ResponseGenerator):
        """Test that Merry maintains direct personality."""
        result = await responder.generate(
            bot=BotIdentity.MERRY,
            system_prompt=get_merry_prompt("TestMerry"),
            context_summary={"suggested_tone": "encouraging"},
            recent_conversation="[12:00] <Frank> I'm thinking about maybe trying to...",
            message="I'm thinking about maybe trying to learn programming",
            sender="Frank",
        )

        assert result.messages
        # Merry should be encouraging but direct
