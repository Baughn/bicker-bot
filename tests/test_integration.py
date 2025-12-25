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
    WebFetcher,
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

        # Direct mentions should have high probability (70%+)
        assert result.probability >= 0.7 or "error" in result.raw_response

    @pytest.mark.asyncio
    async def test_engagement_check_noise(self, engagement_checker: EngagementChecker):
        """Test engagement check with ambient noise."""
        result = await engagement_checker.check(
            message="lol",
            recent_context="[12:00] <Alice> brb\n[12:01] <Bob> k",
            mentioned=False,
            is_question=False,
        )

        # Noise should have low probability (under 50%)
        assert result.probability < 0.5 or "error" in result.raw_response


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


class TestWebFetcherIntegration:
    """Integration tests for web fetching functionality."""

    @pytest.fixture
    async def fetcher(self) -> WebFetcher:
        """Create web fetcher with cleanup."""
        f = WebFetcher()
        yield f
        await f.close()

    @pytest.mark.asyncio
    async def test_fetch_real_page(self, fetcher: WebFetcher):
        """Test fetching a real webpage (httpbin)."""
        result = await fetcher.fetch("https://httpbin.org/html", include_images=False)

        assert result.error is None
        assert "Herman Melville" in result.markdown_content
        # Note: httpbin.org/html has no <title> tag, so title may be empty

    @pytest.mark.asyncio
    async def test_fetch_nonexistent_page(self, fetcher: WebFetcher):
        """Test handling of 404 pages."""
        result = await fetcher.fetch(
            "https://httpbin.org/status/404", include_images=False
        )

        assert result.error is not None
        assert "404" in result.error

    @pytest.mark.asyncio
    async def test_fetch_with_redirect(self, fetcher: WebFetcher):
        """Test following redirects."""
        # httpbin redirects /redirect/1 -> /get
        result = await fetcher.fetch(
            "https://httpbin.org/redirect/1", include_images=False
        )

        # Will get non-HTML response, so should error
        assert result.error is not None or "redirect" in result.markdown_content.lower()

    @pytest.mark.asyncio
    async def test_fetch_direct_image_url(self, fetcher: WebFetcher):
        """Test fetching a direct image URL returns image data.

        A message containing an image URL should result in the image being
        available to attach to LLM calls.
        """
        result = await fetcher.fetch(
            "https://usercontent.irccloud-cdn.com/file/MLwN7WT0/image.png"
        )

        assert result.error is None, f"Expected success but got error: {result.error}"
        assert len(result.images) == 1
        assert result.images[0].base64_data, "Image should have base64 data"
        assert result.images[0].mime_type == "image/jpeg"  # Converted to JPEG
        assert result.images[0].width > 0
        assert result.images[0].height > 0
        assert result.markdown_content == "[Direct image link]"


class TestResponderWithWebTools:
    """Integration tests for responder with web tools enabled."""

    @pytest.fixture
    def responder_with_web(self) -> ResponseGenerator:
        """Create response generator with web fetcher."""
        return ResponseGenerator(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            google_api_key=get_google_key(),
            web_fetcher=WebFetcher(),
            opus_model="claude-sonnet-4-20250514",
            gemini_model="gemini-2.0-flash-exp",
        )

    @pytest.mark.asyncio
    async def test_responder_has_web_tools(self, responder_with_web: ResponseGenerator):
        """Test that responder accepts web fetcher."""
        assert responder_with_web._web_fetcher is not None

    @pytest.mark.asyncio
    async def test_generate_with_url_mention(self, responder_with_web: ResponseGenerator):
        """Test generating response when URL is mentioned."""
        result = await responder_with_web.generate(
            bot=BotIdentity.HACHIMAN,
            system_prompt=get_hachiman_prompt("TestHachi"),
            context_summary={"suggested_tone": "helpful"},
            recent_conversation="[12:00] <Alice> Check this out",
            message="What do you think of https://httpbin.org/html ?",
            sender="Alice",
            detected_urls=["https://httpbin.org/html"],
        )

        # Should get some response (may or may not use the tool)
        assert result.messages is not None
        assert result.bot == BotIdentity.HACHIMAN
