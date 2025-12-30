"""Main orchestrator tying all components together."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn

from bicker_bot.config import Config, get_bot_nicks
from bicker_bot.core import (
    BickerChecker,
    ContextBuilder,
    ConversationStore,
    EngagementChecker,
    MessageRouter,
    ResponseGate,
    ResponseGenerator,
    WebFetcher,
)
from bicker_bot.core.logging import get_session_stats, log_timing
from bicker_bot.debug.server import create_app
from bicker_bot.irc.client import IRCClient, Message, MessageType
from bicker_bot.memory import BotIdentity, BotSelector, MemoryExtractor, MemoryStore
from bicker_bot.personalities import get_personality_prompt
from bicker_bot.tracing import TraceContext, TraceStore

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
    truncated: bool = False  # True if output hit token limit (response discarded)


@dataclass
class ReplayResult:
    """Result of replaying a trace for A/B comparison."""

    original: TraceContext
    replayed: TraceContext
    decision_diffs: list[dict]  # [{stage, original_decision, replayed_decision, diverged}]


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
        self._conversation_store = ConversationStore(Path("data/conversations.db"))
        self._router = MessageRouter(buffer_size=30, store=self._conversation_store)
        self._gate = ResponseGate(config.gate, self._bot_nicks)

        self._memory_store = MemoryStore(config.memory)
        self._bot_selector = BotSelector(config.memory)

        self._engagement = EngagementChecker(google_key)
        self._bicker_checker = BickerChecker(google_key)
        self._web_fetcher = WebFetcher()
        self._context_builder = ContextBuilder(
            google_key, self._memory_store, web_fetcher=self._web_fetcher
        )
        self._responder = ResponseGenerator(
            anthropic_key,
            google_key,
            web_fetcher=self._web_fetcher,
            on_error_notify=self._notify_error,
        )
        self._extractor = MemoryExtractor(
            google_key, self._memory_store, dedup_config=config.memory
        )

        self._irc: IRCClient | None = None
        self._error_notify_channel: str | None = None  # Set after IRC connects

        # Message buffering: wait for 2s of silence before processing
        self._channel_buffers: dict[str, ChannelBuffer] = {}

        # Trace storage for debug observability
        self._trace_store = TraceStore(Path("data/traces.db"))

        logger.info("Orchestrator initialized")

    async def _notify_error(self, message: str) -> None:
        """Send an error notification to IRC (pings Baughn)."""
        if self._irc and self._error_notify_channel:
            try:
                await self._irc.send_as_merry(self._error_notify_channel, message)
            except Exception as e:
                logger.error(f"Failed to send error notification: {e}")

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
        logger.info("Debug server started on http://127.0.0.1:8080")

        # Continue with IRC connection
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
        # Check if this is a bot message - might trigger bickering
        # Don't add to router here; bot messages are added when sent in _send_responses
        if self._irc and self._irc.is_bot_message(message.sender):
            await self._maybe_bicker_response(message)
            return

        # Add human messages to router for context tracking
        await self._router.add_message(message)

        # Log received message
        if message.type == MessageType.ACTION:
            logger.info(f"MSG_RECEIVED: * {message.sender} {message.content}")
        elif message.type == MessageType.MODE_CHANGE:
            logger.info(f"MSG_RECEIVED: ** {message.content}")
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
                channel, responding_bot, result.messages, message, result.truncated
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
                    channel, result.bot, result.messages, representative, result.truncated
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
                await self._send_responses(
                    channel, bot, result.messages, representative, result.truncated
                )

    async def _send_responses(
        self,
        channel: str,
        bot: BotIdentity,
        messages: list[str],
        representative_msg: Message,
        truncated: bool = False,
    ) -> None:
        """Send response messages and handle memory extraction."""
        if not self._irc:
            return

        bot_name = "merry" if bot == BotIdentity.MERRY else "hachiman"
        bot_nick = (
            self._config.irc.nick_merry
            if bot == BotIdentity.MERRY
            else self._config.irc.nick_hachiman
        )

        for i, msg in enumerate(messages):
            # Delay between separate messages (not before first)
            if i > 0:
                await asyncio.sleep(0.33)

            # Check for /me prefix to send as action
            if msg.startswith("/me "):
                action_content = msg[4:]  # Strip "/me "
                await self._irc.send(
                    bot_name, channel, action_content, msg_type=MessageType.ACTION
                )
                # Add bot's message to router so it appears in context
                bot_msg = Message(
                    channel=channel,
                    sender=bot_nick,
                    content=action_content,
                    type=MessageType.ACTION,
                )
            else:
                await self._irc.send(bot_name, channel, msg)
                # Add bot's message to router so it appears in context
                bot_msg = Message(channel=channel, sender=bot_nick, content=msg)
            await self._router.add_message(bot_msg)

        # Record that the bot spoke (for alternation bias)
        if messages:
            self._bot_selector.record_message(bot, messages[0])

        # Extract memories in background (skip if truncated - output was garbage)
        if not truncated:
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

        # Create trace context for this pipeline run
        ctx = TraceContext(
            channel=channel,
            trigger_messages=[combined_content],
            config_snapshot={},  # TODO: Use config loader snapshot
        )

        # Skip gate and engagement for direct addresses
        # Record that we bypassed gate/engagement
        ctx.add_step(
            stage="gate",
            inputs={"message": combined_content},
            outputs={"should_respond": True},
            decision="Direct address bypasses gate",
            details={"forced_bot": forced_bot.value},
        )

        # Extract URLs from message for web fetching
        detected_urls = self._web_fetcher.extract_urls_from_text(combined_content)

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
                detected_urls=detected_urls,
                trace_ctx=ctx,
            )

        # Generate response
        online_users = self._irc.get_channel_users(message.channel) if self._irc else []
        with log_timing(logger, "Response generation (direct)"):
            response_result = await self._responder.generate(
                bot=forced_bot,
                system_prompt=get_personality_prompt(forced_bot, self._config),
                context_summary=context_result.summary,
                recent_conversation=full_context,
                message=combined_content,
                sender=message.sender,
                detected_urls=detected_urls,
                online_users=online_users,
                trace_ctx=ctx,
            )

        pipeline_elapsed = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            f"PIPELINE_COMPLETE (direct): bot={forced_bot.value} "
            f"messages={len(response_result.messages)} total={pipeline_elapsed:.0f}ms"
        )

        # Save trace with final result
        ctx.final_result = response_result.messages if response_result.messages else None
        self._trace_store.save(ctx)

        return ProcessingResult(
            responded=bool(response_result.messages),
            bot=forced_bot,
            messages=response_result.messages,
            gate_passed=True,
            engagement_passed=True,
            reason="Direct address pipeline completed",
            truncated=response_result.truncated,
        )

    async def _process_message(self, message: Message) -> ProcessingResult:
        """Process a message through the full pipeline."""
        pipeline_start = time.perf_counter()
        channel = message.channel

        # Create trace context for this pipeline run
        ctx = TraceContext(
            channel=channel,
            trigger_messages=[message.content],
            config_snapshot={},  # TODO: Use config loader snapshot
        )

        # Get conversation context
        last_activity = await self._router.get_last_activity(channel)
        consecutive_bot = await self._router.count_consecutive_bot_messages(
            channel, self._bot_nicks
        )

        # Step 1: Analyze gate factors
        with log_timing(logger, "Gate check"):
            gate_result = self._gate.should_respond(
                message=message.content,
                last_activity=last_activity,
                consecutive_bot_messages=consecutive_bot,
                is_mode_change=message.type == MessageType.MODE_CHANGE,
                trace_ctx=ctx,
            )

        # Gate is a fast-path bypass: if it passes, skip engagement check
        # If it fails, we still run the engagement check to let the LLM decide
        bypass_engagement = gate_result.should_respond

        if not bypass_engagement:
            # Step 2: Engagement check (LLM-based)
            recent_messages = await self._router.get_recent_messages(channel, count=10)
            recent_context = self._router.format_context(recent_messages)

            with log_timing(logger, "Engagement check"):
                engagement_result = await self._engagement.check(
                    message=message.content,
                    recent_context=recent_context,
                    mentioned=gate_result.factors.mentioned,
                    is_question=gate_result.factors.is_question,
                    trace_ctx=ctx,
                )

            # Roll against engagement probability
            roll = random.random()
            if roll >= engagement_result.probability:
                logger.info(
                    f"ENGAGE: P={engagement_result.probability:.3f} "
                    f"roll={roll:.3f} -> SKIP"
                )
                # Save trace before early exit
                ctx.final_result = None
                self._trace_store.save(ctx)
                return ProcessingResult(
                    responded=False,
                    gate_passed=False,
                    engagement_passed=False,
                    reason=(
                        f"Engagement roll failed: "
                        f"P={engagement_result.probability:.2f} roll={roll:.2f}"
                    ),
                )
            logger.info(
                f"ENGAGE: P={engagement_result.probability:.3f} "
                f"roll={roll:.3f} -> RESPOND"
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

        # Extract URLs from message for web fetching
        detected_urls = self._web_fetcher.extract_urls_from_text(message.content)

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
                detected_urls=detected_urls,
                trace_ctx=ctx,
            )

        # Step 5: Generate response
        online_users = self._irc.get_channel_users(message.channel) if self._irc else []
        with log_timing(logger, "Response generation"):
            response_result = await self._responder.generate(
                bot=selected_bot,
                system_prompt=get_personality_prompt(selected_bot, self._config),
                context_summary=context_result.summary,
                recent_conversation=full_context,
                message=message.content,
                sender=message.sender,
                detected_urls=detected_urls,
                online_users=online_users,
                trace_ctx=ctx,
            )

        # Log pipeline completion
        pipeline_elapsed = (time.perf_counter() - pipeline_start) * 1000
        logger.info(
            f"PIPELINE_COMPLETE: gate=PASS engage=PASS "
            f"bot={selected_bot.value} rounds={context_result.rounds} "
            f"total={pipeline_elapsed:.0f}ms"
        )

        # Save trace with final result
        ctx.final_result = response_result.messages if response_result.messages else None
        self._trace_store.save(ctx)

        return ProcessingResult(
            responded=bool(response_result.messages),
            bot=selected_bot,
            messages=response_result.messages,
            gate_passed=True,
            engagement_passed=True,
            reason="Full pipeline completed",
            truncated=response_result.truncated,
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

    async def replay(
        self, trace_id: str, config_overrides: dict | None = None
    ) -> ReplayResult:
        """Replay a trace with optional config overrides.

        Runs the pipeline in dry-run mode (no IRC, no memory writes).

        Args:
            trace_id: ID of the original trace to replay
            config_overrides: Optional config values to override for replay

        Returns:
            ReplayResult with original trace, replayed trace, and decision diffs

        Raises:
            ValueError: If trace not found
        """
        original = self._trace_store.get(trace_id)
        if original is None:
            raise ValueError(f"Trace {trace_id} not found")

        # Get trigger text from original
        trigger_text = original.trigger_messages[0] if original.trigger_messages else ""

        # Create replay trace context
        replay_ctx = TraceContext(
            channel=original.channel,
            trigger_messages=original.trigger_messages,
            config_snapshot=config_overrides or {},
            is_replay=True,
            original_trace_id=original.id,
        )

        # Build a mock message for the pipeline (kept for potential future use)
        _message = Message(
            channel=original.channel,
            sender="replay",  # placeholder sender
            content=trigger_text,
        )

        # Run pipeline components manually in dry-run mode
        # (Skip IRC sending and memory extraction)

        # Step 1: Gate check
        gate_result = self._gate.should_respond(
            message=trigger_text,
            last_activity=None,  # No activity tracking in replay
            consecutive_bot_messages=0,
            is_mode_change=False,
            trace_ctx=replay_ctx,
        )

        # For replay, continue even if gate fails (to show what would happen)

        # Step 2: Engagement check (if gate didn't pass)
        if not gate_result.should_respond:
            # Use empty context for replay engagement check
            _engagement_result = await self._engagement.check(
                message=trigger_text,
                recent_context="",  # No context available in replay
                mentioned=gate_result.factors.mentioned,
                is_question=gate_result.factors.is_question,
                trace_ctx=replay_ctx,
            )
            # For replay, continue regardless of result

        # Step 3: Bot selection
        if gate_result.factors.directly_addressed and gate_result.factors.addressed_bot:
            nick_lower = gate_result.factors.addressed_bot
            if nick_lower == self._config.irc.nick_merry.lower():
                selected_bot = BotIdentity.MERRY
            else:
                selected_bot = BotIdentity.HACHIMAN
        else:
            selection = self._bot_selector.select(trigger_text)
            selected_bot = selection.selected

        # Record bot selection in replay context
        replay_ctx.add_step(
            stage="selector",
            inputs={"message": trigger_text},
            outputs={"selected": selected_bot.value},
            decision=f"Selected {selected_bot.value}",
        )

        # Step 4: Context building
        bot_nickname = (
            self._config.irc.nick_merry
            if selected_bot == BotIdentity.MERRY
            else self._config.irc.nick_hachiman
        )

        # Get high intensity memories if sender was recorded
        high_intensity: list[str] = []  # Empty for replay without real sender

        # Extract URLs from message
        detected_urls = self._web_fetcher.extract_urls_from_text(trigger_text)

        context_result = await self._context_builder.build(
            message=trigger_text,
            recent_context="",  # No recent context in replay
            sender="replay",
            high_intensity_memories=high_intensity,
            bot_identity=selected_bot,
            bot_nickname=bot_nickname,
            detected_urls=detected_urls,
            trace_ctx=replay_ctx,
        )

        # Step 5: Response generation
        response_result = await self._responder.generate(
            bot=selected_bot,
            system_prompt=get_personality_prompt(selected_bot, self._config),
            context_summary=context_result.summary,
            recent_conversation="",  # No recent conversation in replay
            message=trigger_text,
            sender="replay",
            detected_urls=detected_urls,
            online_users=[],  # No channel context in replay
            trace_ctx=replay_ctx,
        )

        # Set final result
        replay_ctx.final_result = (
            response_result.messages if response_result.messages else None
        )

        # Save replay trace
        self._trace_store.save(replay_ctx)

        # Compare decisions between original and replayed
        decision_diffs = []
        for orig_step in original.steps:
            replay_step = next(
                (s for s in replay_ctx.steps if s.stage == orig_step.stage), None
            )
            if replay_step:
                decision_diffs.append({
                    "stage": orig_step.stage,
                    "original_decision": orig_step.decision,
                    "replayed_decision": replay_step.decision,
                    "diverged": orig_step.decision != replay_step.decision,
                })
            else:
                # Original step not present in replay
                decision_diffs.append({
                    "stage": orig_step.stage,
                    "original_decision": orig_step.decision,
                    "replayed_decision": "(not executed)",
                    "diverged": True,
                })

        # Check for steps in replay not in original
        original_stages = {s.stage for s in original.steps}
        for replay_step in replay_ctx.steps:
            if replay_step.stage not in original_stages:
                decision_diffs.append({
                    "stage": replay_step.stage,
                    "original_decision": "(not executed)",
                    "replayed_decision": replay_step.decision,
                    "diverged": True,
                })

        logger.info(
            f"REPLAY_COMPLETE: trace={trace_id[:8]}... "
            f"diffs={sum(1 for d in decision_diffs if d['diverged'])}/{len(decision_diffs)}"
        )

        return ReplayResult(
            original=original,
            replayed=replay_ctx,
            decision_diffs=decision_diffs,
        )

    @property
    def trace_store(self) -> TraceStore:
        """Expose trace store for debug server."""
        return self._trace_store
