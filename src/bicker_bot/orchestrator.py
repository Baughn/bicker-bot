"""Main orchestrator tying all components together."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field

from bicker_bot.config import Config, get_bot_nicks
from bicker_bot.core import (
    BickerChecker,
    ContextBuilder,
    EngagementChecker,
    MessageRouter,
    ResponseGate,
    ResponseGenerator,
)
from bicker_bot.core.logging import get_session_stats, log_timing
from bicker_bot.irc.client import IRCClient, Message
from bicker_bot.memory import BotIdentity, BotSelector, MemoryExtractor, MemoryStore
from bicker_bot.personalities import get_personality_prompt

logger = logging.getLogger(__name__)

# How often to log session stats (every N messages)
STATS_LOG_INTERVAL = 50


@dataclass
class ProcessingResult:
    """Result of processing a message."""

    responded: bool
    bot: BotIdentity | None = None
    messages: list[str] | None = None  # Can be empty, 1, or multiple
    gate_passed: bool = False
    engagement_passed: bool = False
    reason: str = ""


@dataclass
class ChannelBuffer:
    """Buffer for messages in a single channel."""

    messages: list[Message] = field(default_factory=list)
    timer: asyncio.Task | None = None


# Default buffer delay in seconds
BUFFER_DELAY_SECONDS = 2.0


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
        self._bicker_checker = BickerChecker(google_key)
        self._context_builder = ContextBuilder(google_key, self._memory_store)
        self._responder = ResponseGenerator(
            anthropic_key,
            google_key,
            on_error_notify=self._notify_error,
        )
        self._extractor = MemoryExtractor(google_key, self._memory_store)

        self._irc: IRCClient | None = None
        self._error_notify_channel: str | None = None  # Set after IRC connects

        # Message buffering: wait for 2s of silence before processing
        self._channel_buffers: dict[str, ChannelBuffer] = {}

        logger.info("Orchestrator initialized")

    async def _notify_error(self, message: str) -> None:
        """Send an error notification to IRC (pings Baughn)."""
        if self._irc and self._error_notify_channel:
            try:
                await self._irc.send_as_merry(self._error_notify_channel, message)
            except Exception as e:
                logger.error(f"Failed to send error notification: {e}")

    async def start(self) -> None:
        """Start the IRC connection and begin processing."""
        self._irc = IRCClient(
            config=self._config,
            on_message=self._handle_message,
        )

        logger.info("Connecting to IRC...")
        await self._irc.connect()

        # Set error notify channel to first configured channel
        if self._config.irc.channels:
            self._error_notify_channel = self._config.irc.channels[0]
        logger.info("Connected! Running forever...")
        await self._irc.run_forever()

    async def _handle_message(self, message: Message) -> None:
        """Handle an incoming IRC message.

        Buffers messages and processes after 2 seconds of silence.
        """
        # Always add to router for context tracking
        await self._router.add_message(message)

        # Check if this is a bot message - might trigger bickering
        if self._irc and self._irc.is_bot_message(message.sender):
            await self._maybe_bicker_response(message)
            return

        # Log received message
        if message.is_action:
            logger.info(f"MSG_RECEIVED: * {message.sender} {message.content}")
        else:
            logger.info(f"MSG_RECEIVED: <{message.sender}> {message.content}")

        # Track stats
        stats = get_session_stats()
        stats.increment("messages_received")

        # Periodic stats logging
        if stats.messages_received % STATS_LOG_INTERVAL == 0:
            logger.info(f"SESSION_STATS: {stats.summary_line()}")

        # Add to channel buffer
        channel = message.channel
        if channel not in self._channel_buffers:
            self._channel_buffers[channel] = ChannelBuffer()

        buf = self._channel_buffers[channel]
        buf.messages.append(message)

        # Cancel existing timer and start new one
        if buf.timer:
            buf.timer.cancel()
        buf.timer = asyncio.create_task(self._process_buffer_after_delay(channel))

    async def _process_buffer_after_delay(self, channel: str) -> None:
        """Wait for delay then process buffered messages."""
        try:
            await asyncio.sleep(BUFFER_DELAY_SECONDS)
        except asyncio.CancelledError:
            return  # New message arrived, timer was reset

        buf = self._channel_buffers.get(channel)
        if not buf or not buf.messages:
            return

        # Grab messages and clear buffer
        messages = buf.messages
        buf.messages = []
        buf.timer = None

        logger.info(f"BUFFER_PROCESS: Processing {len(messages)} messages from {channel}")

        # Process the batch
        await self._process_batch(channel, messages)

    async def _maybe_bicker_response(self, message: Message) -> None:
        """Check if a bot message should trigger the other bot to respond.

        Uses Gemini Flash to evaluate if the message warrants a comeback,
        then applies decay based on consecutive bot messages.
        """
        channel = message.channel

        # Get context and consecutive count
        recent = await self._router.get_recent_messages(channel, count=10)
        context = self._router.format_context(recent)
        consecutive = await self._router.count_consecutive_bot_messages(
            channel, self._bot_nicks
        )

        # Ask Flash for probability
        bicker_result = await self._bicker_checker.check(
            message=message.content,
            sender=message.sender,
            recent_context=context,
        )

        # Apply decay
        decay = self._config.gate.decay_factor ** consecutive
        final_prob = bicker_result.probability * decay

        roll = random.random()
        if roll >= final_prob:
            logger.info(
                f"BICKER: base={bicker_result.probability:.2f} decay={decay:.3f} "
                f"final={final_prob:.3f} roll={roll:.3f} -> SKIP"
            )
            return

        logger.info(
            f"BICKER: base={bicker_result.probability:.2f} decay={decay:.3f} "
            f"final={final_prob:.3f} roll={roll:.3f} -> RESPOND"
        )

        # The OTHER bot responds
        if message.sender.lower() == self._config.irc.nick_merry.lower():
            responding_bot = BotIdentity.HACHIMAN
        else:
            responding_bot = BotIdentity.MERRY

        # Generate response (reuse existing pipeline)
        result = await self._process_direct_addressed(
            message, message.content, responding_bot
        )

        if result.responded and result.messages:
            await self._send_responses(
                channel, responding_bot, result.messages, message
            )

    async def _process_batch(self, channel: str, messages: list[Message]) -> None:
        """Process a batch of buffered messages.

        Groups messages by direct address, allowing both bots to respond
        if both are addressed.
        """
        # Partition messages by direct address
        merry_messages: list[Message] = []
        hachi_messages: list[Message] = []
        general_messages: list[Message] = []

        merry_nick = self._config.irc.nick_merry.lower()

        for msg in messages:
            is_direct, bot_nick = self._gate.check_direct_address(msg.content)
            if is_direct and bot_nick:
                if bot_nick == merry_nick:
                    merry_messages.append(msg)
                else:
                    hachi_messages.append(msg)
            else:
                general_messages.append(msg)

        # Collect response tasks: (bot, messages)
        response_tasks: list[tuple[BotIdentity, list[Message]]] = []

        # Direct-addressed messages: bypass gate/engagement, force bot
        if merry_messages:
            response_tasks.append((BotIdentity.MERRY, merry_messages))
        if hachi_messages:
            response_tasks.append((BotIdentity.HACHIMAN, hachi_messages))

        # General messages: run through normal pipeline
        if general_messages:
            # Use the last message as representative for gate/engagement
            representative = general_messages[-1]
            result = await self._process_message(representative)
            if result.responded and result.messages and result.bot:
                await self._send_responses(
                    channel, result.bot, result.messages, representative
                )

        # Randomize order for direct-addressed responses
        random.shuffle(response_tasks)

        # Process each direct-addressed group
        for bot, msgs in response_tasks:
            # Combine messages for context
            combined_content = " | ".join(m.content for m in msgs)
            # Use last message as representative
            representative = msgs[-1]

            # Run through pipeline but with forced bot
            result = await self._process_direct_addressed(
                representative, combined_content, bot
            )
            if result.responded and result.messages:
                await self._send_responses(channel, bot, result.messages, representative)

    async def _send_responses(
        self,
        channel: str,
        bot: BotIdentity,
        messages: list[str],
        representative_msg: Message,
    ) -> None:
        """Send response messages and handle memory extraction."""
        if not self._irc:
            return

        bot_name = "merry" if bot == BotIdentity.MERRY else "hachiman"

        for msg in messages:
            # Check for /me prefix to send as action
            if msg.startswith("/me "):
                action_content = msg[4:]  # Strip "/me "
                await self._irc.send(bot_name, channel, action_content, is_action=True)
            else:
                await self._irc.send(bot_name, channel, msg)

        # Record that the bot spoke (for alternation bias)
        if messages:
            self._bot_selector.record_message(bot, messages[0])

        # Extract memories in background
        all_responses = " ".join(messages)
        asyncio.create_task(
            self._extract_memories(channel, all_responses, bot_name)
        )

    async def _process_direct_addressed(
        self,
        message: Message,
        combined_content: str,
        forced_bot: BotIdentity,
    ) -> ProcessingResult:
        """Process a directly-addressed message with forced bot selection."""
        pipeline_start = time.perf_counter()
        channel = message.channel

        # Skip gate and engagement for direct addresses

        # Build context
        all_recent = await self._router.get_recent_messages(channel, count=30)
        full_context = self._router.format_context(all_recent)

        high_intensity = self._memory_store.get_high_intensity_memories(message.sender)

        bot_nickname = (
            self._config.irc.nick_merry
            if forced_bot == BotIdentity.MERRY
            else self._config.irc.nick_hachiman
        )

        with log_timing(logger, "Context building (direct)"):
            context_result = await self._context_builder.build(
                message=combined_content,
                recent_context=full_context,
                sender=message.sender,
                high_intensity_memories=high_intensity,
                bot_identity=forced_bot,
                bot_nickname=bot_nickname,
            )

        # Generate response
        with log_timing(logger, "Response generation (direct)"):
            response_result = await self._responder.generate(
                bot=forced_bot,
                system_prompt=get_personality_prompt(forced_bot, self._config),
                context_summary=context_result.summary,
                recent_conversation=full_context,
                message=combined_content,
                sender=message.sender,
            )

        pipeline_elapsed = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            f"PIPELINE_COMPLETE (direct): bot={forced_bot.value} "
            f"messages={len(response_result.messages)} total={pipeline_elapsed:.0f}ms"
        )

        return ProcessingResult(
            responded=bool(response_result.messages),
            bot=forced_bot,
            messages=response_result.messages,
            gate_passed=True,
            engagement_passed=True,
            reason="Direct address pipeline completed",
        )

    async def _process_message(self, message: Message) -> ProcessingResult:
        """Process a message through the full pipeline."""
        pipeline_start = time.perf_counter()
        channel = message.channel

        # Get conversation context
        last_activity = await self._router.get_last_activity(channel)
        consecutive_bot = await self._router.count_consecutive_bot_messages(
            channel, self._bot_nicks
        )

        # Step 1: Statistical gate
        with log_timing(logger, "Gate check"):
            gate_result = self._gate.should_respond(
                message=message.content,
                last_activity=last_activity,
                consecutive_bot_messages=consecutive_bot,
            )

        if not gate_result.should_respond:
            return ProcessingResult(
                responded=False,
                gate_passed=False,
                reason=f"Gate failed: P={gate_result.probability:.3f}",
            )

        # Step 2: Engagement check (LLM-based)
        recent_messages = await self._router.get_recent_messages(channel, count=5)
        recent_context = self._router.format_context(recent_messages)

        with log_timing(logger, "Engagement check"):
            engagement_result = await self._engagement.check(
                message=message.content,
                recent_context=recent_context,
                mentioned=gate_result.factors.mentioned,
                is_question=gate_result.factors.is_question,
            )

        # Direct addresses and mentions override engagement check
        bypass_engagement = (
            gate_result.factors.directly_addressed or gate_result.factors.mentioned
        )
        if not engagement_result.is_engaged and not bypass_engagement:
            return ProcessingResult(
                responded=False,
                gate_passed=True,
                engagement_passed=False,
                reason=f"Engagement check failed: {engagement_result.raw_response}",
            )

        # Step 3: Bot selection
        # Direct address forces specific bot
        if gate_result.factors.directly_addressed and gate_result.factors.addressed_bot:
            nick_lower = gate_result.factors.addressed_bot
            if nick_lower == self._config.irc.nick_merry.lower():
                selected_bot = BotIdentity.MERRY
            else:
                selected_bot = BotIdentity.HACHIMAN
            logger.info(f"BOT_SELECT: {selected_bot.value} (direct address)")
        else:
            with log_timing(logger, "Bot selection"):
                selection = self._bot_selector.select(message.content)
                selected_bot = selection.selected

        # Step 4: Build context
        all_recent = await self._router.get_recent_messages(channel, count=30)
        full_context = self._router.format_context(all_recent)

        high_intensity = self._memory_store.get_high_intensity_memories(message.sender)

        # Get bot nickname for context building
        bot_nickname = (
            self._config.irc.nick_merry
            if selected_bot == BotIdentity.MERRY
            else self._config.irc.nick_hachiman
        )

        with log_timing(logger, "Context building"):
            context_result = await self._context_builder.build(
                message=message.content,
                recent_context=full_context,
                sender=message.sender,
                high_intensity_memories=high_intensity,
                bot_identity=selected_bot,
                bot_nickname=bot_nickname,
            )

        # Step 5: Generate response
        with log_timing(logger, "Response generation"):
            response_result = await self._responder.generate(
                bot=selected_bot,
                system_prompt=get_personality_prompt(selected_bot, self._config),
                context_summary=context_result.summary,
                recent_conversation=full_context,
                message=message.content,
                sender=message.sender,
            )

        # Log pipeline completion
        pipeline_elapsed = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            f"PIPELINE_COMPLETE: gate=PASS engage=PASS "
            f"bot={selected_bot.value} rounds={context_result.rounds} "
            f"total={pipeline_elapsed:.0f}ms"
        )

        return ProcessingResult(
            responded=bool(response_result.messages),
            bot=selected_bot,
            messages=response_result.messages,
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
