"""Statistical response gate for deciding whether to respond."""

import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from bicker_bot.config import GateConfig
from bicker_bot.core.logging import get_session_stats

logger = logging.getLogger(__name__)


@dataclass
class GateFactors:
    """Factors that influence response probability."""

    mentioned: bool = False
    is_question: bool = False
    is_conversation_start: bool = False
    consecutive_bot_messages: int = 0

    def __str__(self) -> str:
        parts = []
        if self.mentioned:
            parts.append("mentioned")
        if self.is_question:
            parts.append("question")
        if self.is_conversation_start:
            parts.append("conv_start")
        if self.consecutive_bot_messages > 0:
            parts.append(f"bot_streak={self.consecutive_bot_messages}")
        return f"GateFactors({', '.join(parts) or 'none'})"


@dataclass
class GateResult:
    """Result of the gate decision."""

    should_respond: bool
    probability: float
    factors: GateFactors
    roll: float  # The random value that was compared against probability

    def __str__(self) -> str:
        status = "PASS" if self.should_respond else "FAIL"
        return f"Gate[{status}]: P={self.probability:.3f}, roll={self.roll:.3f}, {self.factors}"


class ResponseGate:
    """Statistically decides whether the bot should respond.

    Uses configurable probability factors:
    - Base probability (always applied)
    - Mention bonus (if bot name mentioned)
    - Question bonus (if message ends with ?)
    - Conversation start bonus (first message after silence)
    - Decay factor (reduces probability for consecutive bot messages)

    Formula: P = min(base + mention + question + conv_start, 1.0) * decay^consecutive_bot_messages
    """

    def __init__(self, config: GateConfig, bot_nicks: tuple[str, str]):
        self._config = config
        self._bot_nicks = tuple(nick.lower() for nick in bot_nicks)
        # Compile patterns for mention detection
        self._mention_patterns = [
            re.compile(rf"\b{re.escape(nick)}\b", re.IGNORECASE) for nick in bot_nicks
        ]

    def _check_mention(self, message: str) -> bool:
        """Check if message mentions either bot."""
        return any(pattern.search(message) for pattern in self._mention_patterns)

    def _check_question(self, message: str) -> bool:
        """Check if message is a question."""
        # Simple heuristic: ends with ? after stripping whitespace
        return message.rstrip().endswith("?")

    def _check_conversation_start(
        self, last_activity: datetime | None, current_time: datetime | None = None
    ) -> bool:
        """Check if this is the first message after a period of silence."""
        if last_activity is None:
            return True  # First message ever

        if current_time is None:
            current_time = datetime.now()

        silence_threshold = timedelta(minutes=self._config.silence_threshold_minutes)
        return (current_time - last_activity) > silence_threshold

    def analyze_factors(
        self,
        message: str,
        last_activity: datetime | None,
        consecutive_bot_messages: int,
        current_time: datetime | None = None,
    ) -> GateFactors:
        """Analyze a message and return the probability factors."""
        return GateFactors(
            mentioned=self._check_mention(message),
            is_question=self._check_question(message),
            is_conversation_start=self._check_conversation_start(last_activity, current_time),
            consecutive_bot_messages=consecutive_bot_messages,
        )

    def calculate_probability(self, factors: GateFactors) -> float:
        """Calculate response probability from factors.

        Returns a value between 0.0 and 1.0.
        """
        cfg = self._config

        # Additive factors (capped at 1.0)
        prob = cfg.base_prob
        if factors.mentioned:
            prob += cfg.mention_prob
        if factors.is_question:
            prob += cfg.question_prob
        if factors.is_conversation_start:
            prob += cfg.conversation_start_prob

        prob = min(prob, 1.0)

        # Multiplicative decay for consecutive bot messages
        # This makes bickering naturally peter out
        if factors.consecutive_bot_messages > 0:
            decay = cfg.decay_factor ** factors.consecutive_bot_messages
            prob *= decay

        return max(0.0, min(1.0, prob))

    def should_respond(
        self,
        message: str,
        last_activity: datetime | None,
        consecutive_bot_messages: int,
        current_time: datetime | None = None,
        _roll: float | None = None,  # For testing
    ) -> GateResult:
        """Decide whether to respond to a message.

        Args:
            message: The message content
            last_activity: Timestamp of last channel activity
            consecutive_bot_messages: Number of consecutive bot messages
            current_time: Current time (for testing)
            _roll: Override random roll (for testing)

        Returns:
            GateResult with decision and metadata
        """
        factors = self.analyze_factors(
            message=message,
            last_activity=last_activity,
            consecutive_bot_messages=consecutive_bot_messages,
            current_time=current_time,
        )

        probability = self.calculate_probability(factors)
        roll = _roll if _roll is not None else random.random()

        result = GateResult(
            should_respond=roll < probability,
            probability=probability,
            factors=factors,
            roll=roll,
        )

        # Log the decision
        status = "PASS" if result.should_respond else "FAIL"
        logger.info(f"GATE: P={probability:.3f} roll={roll:.3f} -> {status} | {factors}")

        # Log detailed calculation at DEBUG
        cfg = self._config
        logger.debug(
            f"Gate calculation: base={cfg.base_prob} + "
            f"mention={cfg.mention_prob if factors.mentioned else 0} + "
            f"question={cfg.question_prob if factors.is_question else 0} + "
            f"conv_start={cfg.conversation_start_prob if factors.is_conversation_start else 0} "
            f"* decay^{factors.consecutive_bot_messages}"
        )

        # Track stats
        stats = get_session_stats()
        if result.should_respond:
            stats.increment("gate_passes")
        else:
            stats.increment("gate_fails")

        return result
