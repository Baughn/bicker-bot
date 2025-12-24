"""Response generator using Opus or Gemini Pro."""

import logging
from dataclasses import dataclass
from typing import Any

import anthropic
from google import genai
from google.genai import types

from bicker_bot.memory import BotIdentity

logger = logging.getLogger(__name__)


@dataclass
class ResponseResult:
    """Result of response generation."""

    content: str
    bot: BotIdentity
    model_used: str


class ResponseGenerator:
    """Generates responses using either Claude Opus or Gemini Pro."""

    def __init__(
        self,
        anthropic_api_key: str,
        google_api_key: str,
        opus_model: str = "claude-opus-4-5-20251101",
        gemini_model: str = "gemini-3-pro",
    ):
        """Initialize the response generator.

        Args:
            anthropic_api_key: Anthropic API key
            google_api_key: Google AI API key
            opus_model: Claude model for Hachiman
            gemini_model: Gemini model for Merry
        """
        self._anthropic = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        self._gemini = genai.Client(api_key=google_api_key)
        self._opus_model = opus_model
        self._gemini_model = gemini_model

    async def generate(
        self,
        bot: BotIdentity,
        system_prompt: str,
        context_summary: dict[str, Any],
        recent_conversation: str,
        message: str,
        sender: str,
    ) -> ResponseResult:
        """Generate a response as the specified bot.

        Args:
            bot: Which bot should respond
            system_prompt: The bot's personality prompt
            context_summary: Summary from context builder
            recent_conversation: Recent conversation formatted
            message: The message to respond to
            sender: Who sent the message

        Returns:
            ResponseResult with the generated content
        """
        # Format the context for the model
        context_str = self._format_context(context_summary)

        user_prompt = f"""Recent conversation:
{recent_conversation}

Context gathered:
{context_str}

Latest message from {sender}: "{message}"

Respond naturally as your character. Keep it conversational and IRC-appropriate (not too long).
Do not use markdown formatting. Do not prefix your response with your name."""

        if bot == BotIdentity.HACHIMAN:
            return await self._generate_opus(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        else:
            return await self._generate_gemini(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context summary for the prompt."""
        lines = []

        if context.get("key_facts"):
            lines.append("Key facts:")
            for fact in context["key_facts"]:
                lines.append(f"  - {fact}")

        if context.get("user_context"):
            lines.append(f"About the user: {context['user_context']}")

        if context.get("topic_context"):
            lines.append(f"Topic background: {context['topic_context']}")

        if context.get("suggested_tone"):
            lines.append(f"Suggested tone: {context['suggested_tone']}")

        return "\n".join(lines) if lines else "No additional context."

    async def _generate_opus(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseResult:
        """Generate a response using Claude Opus."""
        try:
            response = await self._anthropic.messages.create(
                model=self._opus_model,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            content = response.content[0].text

            logger.debug(f"Opus response: {content[:100]}...")

            return ResponseResult(
                content=content,
                bot=BotIdentity.HACHIMAN,
                model_used=self._opus_model,
            )

        except Exception as e:
            logger.error(f"Opus generation failed: {e}")
            return ResponseResult(
                content="...I had a thought, but it got away from me.",
                bot=BotIdentity.HACHIMAN,
                model_used="fallback",
            )

    async def _generate_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseResult:
        """Generate a response using Gemini Pro."""
        try:
            response = await self._gemini.aio.models.generate_content(
                model=self._gemini_model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.8,  # Slightly more creative for Merry
                    max_output_tokens=500,
                ),
            )

            content = response.text

            logger.debug(f"Gemini response: {content[:100]}...")

            return ResponseResult(
                content=content,
                bot=BotIdentity.MERRY,
                model_used=self._gemini_model,
            )

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return ResponseResult(
                content="Tch... something's off. Let me think about this.",
                bot=BotIdentity.MERRY,
                model_used="fallback",
            )
