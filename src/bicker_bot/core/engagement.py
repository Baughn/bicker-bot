"""Engagement check using Gemini Flash."""

import logging
from dataclasses import dataclass

from google import genai
from google.genai import types

from bicker_bot.core.logging import get_session_stats, log_llm_call, log_llm_response

logger = logging.getLogger(__name__)


ENGAGEMENT_SYSTEM_PROMPT = """You are an engagement detector for an IRC chatbot system.
Your job is to determine if a message represents genuine human engagement that warrants a bot response.

Consider these factors:
1. Is the message directed at the conversation or just ambient chatter?
2. Does the message invite response (questions, opinions sought, topics opened)?
3. Is this part of an active discussion or random noise?
4. Would a response feel natural and welcome, or intrusive?

Important: The bots should NOT overwhelm human conversation. When in doubt, lean towards "no".

Respond with exactly one word: "yes" or "no"
"""


@dataclass
class EngagementResult:
    """Result of engagement check."""

    is_engaged: bool
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

Should the bots respond to this? (yes/no)"""

        try:
            # Log LLM input
            log_llm_call(
                operation="Engagement Check",
                model=self._model,
                system_prompt=ENGAGEMENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                config={"temperature": 0.1, "max_output_tokens": 10},
            )

            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=ENGAGEMENT_SYSTEM_PROMPT,
                    temperature=0.1,  # Low temperature for consistent decisions
                    max_output_tokens=10,
                ),
            )

            raw = response.text.strip().lower()

            # Log LLM response
            log_llm_response(
                operation="Engagement Check",
                response_text=raw,
            )
            is_engaged = raw.startswith("yes")

            # Log the decision
            preview = message[:80] + "..." if len(message) > 80 else message
            status = "YES" if is_engaged else "NO"
            logger.info(
                f"ENGAGEMENT: '{preview}' -> {status} "
                f"[mentioned={mentioned}, question={is_question}]"
            )

            # Track stats
            stats = get_session_stats()
            stats.increment_api_call(self._model)
            if is_engaged:
                stats.increment("engagement_passes")
            else:
                stats.increment("engagement_fails")

            return EngagementResult(
                is_engaged=is_engaged,
                raw_response=raw,
            )

        except Exception as e:
            logger.error(f"Engagement check failed: {e}")
            # On error, default to responding if mentioned
            return EngagementResult(
                is_engaged=mentioned,
                raw_response=f"error: {e}",
            )
