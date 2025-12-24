"""Main orchestrator tying all components together."""

import asyncio
import logging
from dataclasses import dataclass

from bicker_bot.config import Config, get_bot_nicks
from bicker_bot.core import (
    ContextBuilder,
    EngagementChecker,
    MessageRouter,
    ResponseGate,
    ResponseGenerator,
)
from bicker_bot.irc.client import IRCClient, Message
from bicker_bot.memory import BotIdentity, BotSelector, MemoryExtractor, MemoryStore
from bicker_bot.personalities import PERSONALITY_PROMPTS

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a message."""

    responded: bool
    bot: BotIdentity | None = None
    response: str | None = None
    gate_passed: bool = False
    engagement_passed: bool = False
    reason: str = ""


class Orchestrator:
    """Orchestrates the entire message processing pipeline."""

    def __init__(self, config: Config):
        """Initialize the orchestrator with all components.

        Args:
            config: Application configuration
        """
        self._config = config
        self._bot_nicks = get_bot_nicks(config)

        # Get API keys
        anthropic_key = (
            config.llm.anthropic_api_key.get_secret_value()
            if config.llm.anthropic_api_key
            else ""
        )
        google_key = (
            config.llm.google_api_key.get_secret_value()
            if config.llm.google_api_key
            else ""
        )

        if not anthropic_key or not google_key:
            raise ValueError("Both ANTHROPIC_API_KEY and GOOGLE_API_KEY must be set")

        # Initialize components
        self._router = MessageRouter(buffer_size=30)
        self._gate = ResponseGate(config.gate, self._bot_nicks)

        self._memory_store = MemoryStore(config.memory)
        self._bot_selector = BotSelector(config.memory)

        self._engagement = EngagementChecker(google_key)
        self._context_builder = ContextBuilder(google_key, self._memory_store)
        self._responder = ResponseGenerator(anthropic_key, google_key)
        self._extractor = MemoryExtractor(google_key, self._memory_store)

        self._irc: IRCClient | None = None

        logger.info("Orchestrator initialized")

    async def start(self) -> None:
        """Start the IRC connection and begin processing."""
        self._irc = IRCClient(
            config=self._config,
            on_message=self._handle_message,
        )

        logger.info("Connecting to IRC...")
        await self._irc.connect()
        logger.info("Connected! Running forever...")
        await self._irc.run_forever()

    async def _handle_message(self, message: Message) -> None:
        """Handle an incoming IRC message.

        This is the main entry point for the processing pipeline.
        """
        # Ignore messages from our own bots
        if self._irc and self._irc.is_bot_message(message.sender):
            # Still add to buffer for context
            await self._router.add_message(message)
            return

        # Add message to buffer
        await self._router.add_message(message)

        # Process the message
        result = await self._process_message(message)

        # Send response if any
        if result.responded and result.response and self._irc:
            bot_name = "merry" if result.bot == BotIdentity.MERRY else "hachiman"
            await self._irc.send(bot_name, message.channel, result.response)

            # Record that the bot spoke (for alternation bias)
            if result.bot:
                self._bot_selector.record_message(result.bot, result.response)

            # Extract memories in background
            asyncio.create_task(
                self._extract_memories(message.channel, result.response, bot_name)
            )

    async def _process_message(self, message: Message) -> ProcessingResult:
        """Process a message through the full pipeline."""
        channel = message.channel

        # Get conversation context
        last_activity = await self._router.get_last_activity(channel)
        consecutive_bot = await self._router.count_consecutive_bot_messages(
            channel, self._bot_nicks
        )

        # Step 1: Statistical gate
        gate_result = self._gate.should_respond(
            message=message.content,
            last_activity=last_activity,
            consecutive_bot_messages=consecutive_bot,
        )

        logger.debug(f"Gate result: {gate_result}")

        if not gate_result.should_respond:
            return ProcessingResult(
                responded=False,
                gate_passed=False,
                reason=f"Gate failed: P={gate_result.probability:.3f}",
            )

        # Step 2: Engagement check (LLM-based)
        recent_messages = await self._router.get_recent_messages(channel, count=5)
        recent_context = self._router.format_context(recent_messages)

        engagement_result = await self._engagement.check(
            message=message.content,
            recent_context=recent_context,
            mentioned=gate_result.factors.mentioned,
            is_question=gate_result.factors.is_question,
        )

        logger.debug(f"Engagement result: {engagement_result}")

        # Mentions override engagement check
        if not engagement_result.is_engaged and not gate_result.factors.mentioned:
            return ProcessingResult(
                responded=False,
                gate_passed=True,
                engagement_passed=False,
                reason=f"Engagement check failed: {engagement_result.raw_response}",
            )

        # Step 3: Bot selection
        selection = self._bot_selector.select(message.content)
        selected_bot = selection.selected

        logger.debug(f"Bot selected: {selected_bot.value} ({selection.reason})")

        # Step 4: Build context
        all_recent = await self._router.get_recent_messages(channel, count=30)
        full_context = self._router.format_context(all_recent)

        high_intensity = self._memory_store.get_high_intensity_memories(message.sender)

        context_result = await self._context_builder.build(
            message=message.content,
            recent_context=full_context,
            sender=message.sender,
            high_intensity_memories=high_intensity,
        )

        logger.debug(f"Context gathered in {context_result.rounds} rounds")

        # Step 5: Generate response
        response_result = await self._responder.generate(
            bot=selected_bot,
            system_prompt=PERSONALITY_PROMPTS[selected_bot],
            context_summary=context_result.summary,
            recent_conversation=full_context,
            message=message.content,
            sender=message.sender,
        )

        logger.info(
            f"[{selected_bot.value}] Response: {response_result.content[:100]}..."
        )

        return ProcessingResult(
            responded=True,
            bot=selected_bot,
            response=response_result.content,
            gate_passed=True,
            engagement_passed=True,
            reason="Full pipeline completed",
        )

    async def _extract_memories(
        self,
        channel: str,
        response: str,
        bot_name: str,
    ) -> None:
        """Extract and store memories from the conversation."""
        try:
            recent = await self._router.get_recent_messages(channel, count=10)
            context = self._router.format_context(recent)

            await self._extractor.extract_and_store(
                conversation=context,
                response_given=response,
                bot_name=bot_name,
            )
        except Exception as e:
            logger.error(f"Memory extraction failed: {e}")
