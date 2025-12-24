"""Tests for the statistical response gate."""

from datetime import datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bicker_bot.config import GateConfig
from bicker_bot.core.gate import GateFactors, GateResult, ResponseGate


class TestGateFactors:
    """Tests for GateFactors dataclass."""

    def test_default_factors(self):
        """Test default factor values."""
        factors = GateFactors()
        assert factors.mentioned is False
        assert factors.is_question is False
        assert factors.is_conversation_start is False
        assert factors.consecutive_bot_messages == 0

    def test_str_representation(self):
        """Test string representation."""
        factors = GateFactors(mentioned=True, is_question=True)
        assert "mentioned" in str(factors)
        assert "question" in str(factors)


class TestResponseGate:
    """Tests for ResponseGate."""

    def test_mention_detection(self, gate_config: GateConfig, bot_nicks: tuple[str, str]):
        """Test that mentions are detected correctly."""
        gate = ResponseGate(gate_config, bot_nicks)

        # Direct mention
        factors = gate.analyze_factors("Hey Merry, what's up?", None, 0)
        assert factors.mentioned is True

        # Case insensitive
        factors = gate.analyze_factors("MERRY help me", None, 0)
        assert factors.mentioned is True

        # Hachi mention
        factors = gate.analyze_factors("Hachi knows", None, 0)
        assert factors.mentioned is True

        # No mention
        factors = gate.analyze_factors("Hello everyone", None, 0)
        assert factors.mentioned is False

        # Partial match should not trigger (e.g., "Merrything")
        factors = gate.analyze_factors("Merrything is fine", None, 0)
        assert factors.mentioned is False

    def test_question_detection(self, gate_config: GateConfig, bot_nicks: tuple[str, str]):
        """Test that questions are detected."""
        gate = ResponseGate(gate_config, bot_nicks)

        # Question mark at end
        factors = gate.analyze_factors("What time is it?", None, 0)
        assert factors.is_question is True

        # Question with trailing space
        factors = gate.analyze_factors("Really?  ", None, 0)
        assert factors.is_question is True

        # Not a question
        factors = gate.analyze_factors("I wonder what time it is", None, 0)
        assert factors.is_question is False

    def test_conversation_start_detection(
        self, gate_config: GateConfig, bot_nicks: tuple[str, str]
    ):
        """Test conversation start detection."""
        gate = ResponseGate(gate_config, bot_nicks)
        now = datetime.now()

        # No previous activity = conversation start
        factors = gate.analyze_factors("Hello", None, 0, current_time=now)
        assert factors.is_conversation_start is True

        # Recent activity = not conversation start
        recent = now - timedelta(minutes=2)
        factors = gate.analyze_factors("Hello", recent, 0, current_time=now)
        assert factors.is_conversation_start is False

        # Old activity = conversation start
        old = now - timedelta(minutes=10)
        factors = gate.analyze_factors("Hello", old, 0, current_time=now)
        assert factors.is_conversation_start is True

    def test_probability_calculation_base(
        self, gate_config: GateConfig, bot_nicks: tuple[str, str]
    ):
        """Test base probability calculation."""
        gate = ResponseGate(gate_config, bot_nicks)

        # Just base probability
        factors = GateFactors()
        prob = gate.calculate_probability(factors)
        assert prob == gate_config.base_prob

    def test_probability_additive_factors(
        self, gate_config: GateConfig, bot_nicks: tuple[str, str]
    ):
        """Test that factors add to probability."""
        gate = ResponseGate(gate_config, bot_nicks)

        # Mention adds
        factors = GateFactors(mentioned=True)
        prob = gate.calculate_probability(factors)
        assert prob == gate_config.base_prob + gate_config.mention_prob

        # Question adds
        factors = GateFactors(is_question=True)
        prob = gate.calculate_probability(factors)
        assert prob == gate_config.base_prob + gate_config.question_prob

        # Multiple factors add (capped at 1.0)
        factors = GateFactors(mentioned=True, is_question=True, is_conversation_start=True)
        prob = gate.calculate_probability(factors)
        expected = min(
            gate_config.base_prob
            + gate_config.mention_prob
            + gate_config.question_prob
            + gate_config.conversation_start_prob,
            1.0,
        )
        assert prob == expected

    def test_decay_factor(self, gate_config: GateConfig, bot_nicks: tuple[str, str]):
        """Test that consecutive bot messages apply decay."""
        gate = ResponseGate(gate_config, bot_nicks)

        # No decay with 0 consecutive
        factors = GateFactors(mentioned=True, consecutive_bot_messages=0)
        prob_0 = gate.calculate_probability(factors)

        # Decay with 1 consecutive
        factors = GateFactors(mentioned=True, consecutive_bot_messages=1)
        prob_1 = gate.calculate_probability(factors)
        assert prob_1 == prob_0 * gate_config.decay_factor

        # More decay with 2 consecutive
        factors = GateFactors(mentioned=True, consecutive_bot_messages=2)
        prob_2 = gate.calculate_probability(factors)
        assert prob_2 == prob_0 * (gate_config.decay_factor**2)

    def test_should_respond_deterministic(
        self, gate_config: GateConfig, bot_nicks: tuple[str, str]
    ):
        """Test should_respond with deterministic roll."""
        gate = ResponseGate(gate_config, bot_nicks)

        # Low roll = should respond
        result = gate.should_respond("Hey Merry?", None, 0, _roll=0.01)
        assert result.should_respond is True

        # High roll = should not respond
        result = gate.should_respond("Random message", None, 0, _roll=0.99)
        assert result.should_respond is False


class TestGateProperties:
    """Property-based tests for the gate."""

    @given(
        base=st.floats(min_value=0.0, max_value=1.0),
        mention=st.floats(min_value=0.0, max_value=1.0),
        question=st.floats(min_value=0.0, max_value=1.0),
        conv_start=st.floats(min_value=0.0, max_value=1.0),
        decay=st.floats(min_value=0.0, max_value=1.0),
        consecutive=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=200)
    def test_probability_always_valid(
        self,
        base: float,
        mention: float,
        question: float,
        conv_start: float,
        decay: float,
        consecutive: int,
    ):
        """Property: probability is always between 0 and 1."""
        config = GateConfig(
            base_prob=base,
            mention_prob=mention,
            question_prob=question,
            conversation_start_prob=conv_start,
            decay_factor=decay,
        )
        gate = ResponseGate(config, ("Bot1", "Bot2"))

        factors = GateFactors(
            mentioned=True,
            is_question=True,
            is_conversation_start=True,
            consecutive_bot_messages=consecutive,
        )

        prob = gate.calculate_probability(factors)
        assert 0.0 <= prob <= 1.0

    @given(consecutive=st.integers(min_value=0, max_value=20))
    def test_decay_monotonic(self, consecutive: int):
        """Property: more consecutive bot messages = lower probability."""
        config = GateConfig(
            base_prob=0.1,
            mention_prob=0.8,
            question_prob=0.3,
            conversation_start_prob=0.4,
            decay_factor=0.5,
        )
        gate = ResponseGate(config, ("Bot1", "Bot2"))

        factors_low = GateFactors(mentioned=True, consecutive_bot_messages=consecutive)
        factors_high = GateFactors(mentioned=True, consecutive_bot_messages=consecutive + 1)

        prob_low = gate.calculate_probability(factors_low)
        prob_high = gate.calculate_probability(factors_high)

        # More consecutive = lower or equal probability
        assert prob_high <= prob_low

    @given(
        mentioned=st.booleans(),
        is_question=st.booleans(),
        is_conv_start=st.booleans(),
    )
    def test_factors_only_increase_probability(
        self,
        mentioned: bool,
        is_question: bool,
        is_conv_start: bool,
    ):
        """Property: enabling any factor should not decrease probability."""
        config = GateConfig(
            base_prob=0.1,
            mention_prob=0.8,
            question_prob=0.3,
            conversation_start_prob=0.4,
            decay_factor=0.5,
        )
        gate = ResponseGate(config, ("Bot1", "Bot2"))

        # Base factors (all False)
        base_factors = GateFactors()
        base_prob = gate.calculate_probability(base_factors)

        # With one or more factors enabled
        test_factors = GateFactors(
            mentioned=mentioned,
            is_question=is_question,
            is_conversation_start=is_conv_start,
        )
        test_prob = gate.calculate_probability(test_factors)

        assert test_prob >= base_prob

    @given(roll=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    def test_gate_decision_consistent(self, roll: float):
        """Property: gate decision is consistent with probability and roll."""
        config = GateConfig(
            base_prob=0.1,
            mention_prob=0.8,
            question_prob=0.3,
            conversation_start_prob=0.4,
            decay_factor=0.5,
        )
        gate = ResponseGate(config, ("Bot1", "Bot2"))

        result = gate.should_respond("Test message", None, 0, _roll=roll)

        # Decision should match roll vs probability comparison
        if roll < result.probability:
            assert result.should_respond is True
        else:
            assert result.should_respond is False
