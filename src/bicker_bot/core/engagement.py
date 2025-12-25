"""Engagement check using Gemini Flash."""

import logging
import re
from dataclasses import dataclass

from google import genai
from google.genai import types
from pydantic import BaseModel

from bicker_bot.core.logging import get_session_stats, log_llm_call, log_llm_response, log_llm_round

logger = logging.getLogger(__name__)


class EngagementResponse(BaseModel):
    """Schema for engagement check response."""

    probability: int  # 0-100


ENGAGEMENT_SYSTEM_PROMPT = """You evaluate whether an IRC message warrants a chatbot response.

Consider:
1. Is the message directed at the bots or the conversation generally?
2. Does it invite response (questions, opinions sought, topics opened)?
3. Is this part of an active discussion or random noise?
4. Would a response feel natural and welcome, or intrusive?

Respond with raw JSON containing a probability 0-100:
- 95-100: Direct interaction (mentions bot by name, asks question to bots, requests help, directly responding to the bot)
- 70-90: Engaging discussion regarding the bot, or a continuation of previous conversation with the bot
- 20-40: Neutral conversation, bots could contribute if relevant
- 5-15: Ambient chatter, bots probably shouldn't jump in
- 0-5: Private conversation or noise, bots should stay out

Example: {"probability": 75}
"""



@dataclass
class EngagementResult:
    """Result of engagement check."""

    probability: float  # 0.0 to 1.0
    raw_response: str


class EngagementChecker:
    """Checks if a message represents genuine human engagement."""

    def __init__(self, api_key: str, model: str = "gemini-3-flash-preview"):
        """Initialize the engagement checker.

        Args:
            api_key: Google AI API key
            model: Model to use (default: gemini-3-flash)
        """
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def check(
        self,
        message: str,
        recent_context: str,
        mentioned: bool = False,
        is_question: bool = False,
    ) -> EngagementResult:
        """Check if a message warrants engagement.

        Args:
            message: The message to check
            recent_context: Recent conversation context (formatted)
            mentioned: Whether a bot was mentioned
            is_question: Whether message is a question

        Returns:
            EngagementResult with decision
        """
        # Build the prompt
        factors = []
        if mentioned:
            factors.append("A bot was directly mentioned.")
        if is_question:
            factors.append("The message is a question.")

        factors_text = " ".join(factors) if factors else "No special factors detected."

        user_prompt = f"""Recent conversation:
{recent_context}

Latest message to evaluate: "{message}"

Detected factors: {factors_text}

What is the probability (0-100) that the bots should respond?"""

        try:
            # Log LLM input
            log_llm_call(
                operation="Engagement Check",
                model=self._model,
                system_prompt=ENGAGEMENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                config={"temperature": 0.1, "max_output_tokens": 512, "json_mode": True},
            )

            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=ENGAGEMENT_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                    response_json_schema=EngagementResponse.model_json_schema(),
                ),
            )

            # Log round summary (always on)
            usage = response.usage_metadata if response else None
            log_llm_round(
                component="engagement",
                model=self._model,
                round_num=1,
                tokens_in=usage.prompt_token_count if usage else None,
                tokens_out=usage.candidates_token_count if usage else None,
            )

            raw = response.text or ""

            # Log LLM response
            log_llm_response(
                operation="Engagement Check",
                response_text=raw,
            )

            # Parse probability from response
            probability = self._parse_probability(raw, mentioned)

            # Log the decision
            preview = message[:80] + "..." if len(message) > 80 else message
            logger.info(
                f"ENGAGEMENT: '{preview}' -> {probability:.0%} "
                f"[mentioned={mentioned}, question={is_question}]"
            )

            # Track stats
            stats = get_session_stats()
            stats.increment_api_call(self._model)

            return EngagementResult(
                probability=probability,
                raw_response=raw,
            )

        except Exception as e:
            logger.error(f"Engagement check failed: {e}")
            # On error, default to high probability if mentioned, low otherwise
            fallback_prob = 0.9 if mentioned else 0.1
            return EngagementResult(
                probability=fallback_prob,
                raw_response=f"error: {e}",
            )

    def _parse_probability(self, raw: str, mentioned: bool) -> float:
        """Parse a probability from the LLM response.

        Args:
            raw: Raw response text (should contain a number 0-100 or JSON)
            mentioned: Whether bot was mentioned (for fallback)

        Returns:
            Probability as float 0.0-1.0
        """
        # Try to parse as JSON first
        try:
            parsed = EngagementResponse.model_validate_json(raw)
            return min(100, max(0, parsed.probability)) / 100.0
        except Exception:
            pass

        # Fall back to extracting first number
        match = re.search(r"\d+", raw)
        if match:
            value = int(match.group())
            return min(100, max(0, value)) / 100.0

        # If we can't parse, return based on mention status
        logger.warning(f"Could not parse engagement probability from: {raw}")
        return 0.9 if mentioned else 0.1
