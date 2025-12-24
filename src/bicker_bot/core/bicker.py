"""Bicker check for bot-to-bot responses using Gemini Flash."""

import logging
import re
from dataclasses import dataclass

from google import genai
from google.genai import types

from bicker_bot.core.logging import get_session_stats, log_llm_call, log_llm_response

logger = logging.getLogger(__name__)


BICKER_SYSTEM_PROMPT = """You evaluate whether a chatbot's message warrants a response from its sibling bot.

The bots are Merry (energetic, action-oriented dream demon) and Hachiman (cynical, analytical loner). They bicker like siblings.

Consider:
1. Is the message teasing, insulting, or challenging the other bot?
2. Does it make a claim the other would want to dispute?
3. Is there an opening for witty sibling banter?
4. Would ignoring this feel unnatural for the sibling dynamic?

Respond with a single integer 0-100 representing the probability the other bot should respond.
- 0-20: Neutral statement, no response needed
- 30-50: Mild opening, might warrant a response
- 60-80: Clear teasing or challenge, likely deserves a comeback
- 90-100: Direct insult or provocation, almost certainly needs a response

Just respond with the number, nothing else.
"""


@dataclass
class BickerResult:
    """Result of bicker check."""

    probability: float  # 0.0 to 1.0
    raw_response: str


class BickerChecker:
    """Checks if a bot message warrants a response from the other bot."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        """Initialize the bicker checker.

        Args:
            api_key: Google AI API key
            model: Model to use (default: gemini-2.0-flash)
        """
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def check(
        self,
        message: str,
        sender: str,
        recent_context: str,
    ) -> BickerResult:
        """Check if a bot message warrants a bicker response.

        Args:
            message: The bot's message to evaluate
            sender: Which bot sent the message (nickname)
            recent_context: Recent conversation context (formatted)

        Returns:
            BickerResult with probability (0.0-1.0)
        """
        user_prompt = f"""Recent conversation:
{recent_context}

The message from {sender}: "{message}"

How likely should the other bot respond? (0-100)"""

        try:
            log_llm_call(
                operation="Bicker Check",
                model=self._model,
                system_prompt=BICKER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                config={"temperature": 0.3, "max_output_tokens": 10},
            )

            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=BICKER_SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=10,
                ),
            )

            raw = response.text.strip()
            log_llm_response(
                operation="Bicker Check",
                response_text=raw,
            )

            # Parse the number
            probability = self._parse_probability(raw)

            logger.info(
                f"BICKER_CHECK: '{message[:60]}...' from {sender} -> {probability:.0%}"
            )

            stats = get_session_stats()
            stats.increment_api_call(self._model)

            return BickerResult(
                probability=probability,
                raw_response=raw,
            )

        except Exception as e:
            logger.error(f"Bicker check failed: {e}")
            # On error, default to low probability
            return BickerResult(
                probability=0.1,
                raw_response=f"error: {e}",
            )

    def _parse_probability(self, raw: str) -> float:
        """Parse a probability from the LLM response.

        Args:
            raw: Raw response text (should be a number 0-100)

        Returns:
            Probability as float 0.0-1.0
        """
        # Extract first number from response
        match = re.search(r"\d+", raw)
        if match:
            value = int(match.group())
            # Clamp to 0-100 and convert to 0.0-1.0
            return min(100, max(0, value)) / 100.0

        # If we can't parse, return low probability
        logger.warning(f"Could not parse bicker probability from: {raw}")
        return 0.1
