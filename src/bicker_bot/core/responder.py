"""Response generator using Opus or Gemini Pro."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic
from google import genai
from google.genai import types

from bicker_bot.core.logging import get_session_stats, log_llm_call, log_llm_response
from bicker_bot.memory import BotIdentity

logger = logging.getLogger(__name__)


@dataclass
class ResponseResult:
    """Result of response generation."""

    messages: list[str] = field(default_factory=list)  # Can be empty, 1, or multiple
    bot: BotIdentity = BotIdentity.HACHIMAN
    model_used: str = ""


class ResponseGenerator:
    """Generates responses using either Claude Opus or Gemini Pro."""

    def __init__(
        self,
        anthropic_api_key: str,
        google_api_key: str,
        opus_model: str = "claude-opus-4-5-20251101",
        gemini_model: str = "gemini-3-pro-preview",
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
Do not use markdown formatting. Do not prefix your response with your name.

You can respond with:
- No messages (if nothing worth saying)
- One message (typical case)
- Multiple messages (if addressing multiple people or topics)

Bias toward fewer messages unless specifically addressing multiple people.

Format your response as a JSON array of strings, like:
["first message"]
or for multiple: ["first message", "second message"]
or for no response: []"""

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

    def _parse_response_json(self, raw: str | None) -> list[str]:
        """Parse LLM response as JSON array of messages."""
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(m) for m in parsed if m]
            # Single string returned instead of array
            return [str(parsed)] if parsed else []
        except json.JSONDecodeError:
            # Fallback: treat raw text as single message
            return [raw.strip()] if raw.strip() else []

    async def _generate_opus(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseResult:
        """Generate a response using Claude Opus."""
        try:
            # Log LLM input
            log_llm_call(
                operation="Response Generation (Hachiman)",
                model=self._opus_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                config={"max_tokens": 500},
            )

            response = await self._anthropic.messages.create(
                model=self._opus_model,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            raw_content = response.content[0].text if response.content else None
            messages = self._parse_response_json(raw_content)

            # Log LLM response
            log_llm_response(
                operation="Response Generation (Hachiman)",
                response_text=raw_content,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )

            # Log response with token usage
            logger.info(
                f"RESPONSE [hachiman]: model={self._opus_model} "
                f"tokens_in={response.usage.input_tokens} "
                f"tokens_out={response.usage.output_tokens} "
                f"messages={len(messages)}"
            )
            logger.info(f"LITERAL_RESPONSE: {raw_content}")

            # Track stats
            stats = get_session_stats()
            stats.increment("responses_hachiman")
            stats.increment_api_call(self._opus_model)

            return ResponseResult(
                messages=messages,
                bot=BotIdentity.HACHIMAN,
                model_used=self._opus_model,
            )

        except Exception as e:
            logger.error(f"Opus generation failed: {e}")
            logger.warning("RESPONSE [hachiman]: Using fallback response")
            return ResponseResult(
                messages=["...I had a thought, but it got away from me."],
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
            # Log LLM input
            log_llm_call(
                operation="Response Generation (Merry)",
                model=self._gemini_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                config={"temperature": 0.8, "max_output_tokens": 500},
            )

            response = await self._gemini.aio.models.generate_content(
                model=self._gemini_model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.8,  # Slightly more creative for Merry
                    max_output_tokens=500,
                ),
            )

            raw_content = response.text if response else None
            messages = self._parse_response_json(raw_content)

            # Log LLM response
            log_llm_response(
                operation="Response Generation (Merry)",
                response_text=raw_content,
            )

            # Log response (Gemini doesn't expose token counts the same way)
            logger.info(f"RESPONSE [merry]: model={self._gemini_model} messages={len(messages)}")
            logger.info(f"LITERAL_RESPONSE: {raw_content}")

            # Track stats
            stats = get_session_stats()
            stats.increment("responses_merry")
            stats.increment_api_call(self._gemini_model)

            return ResponseResult(
                messages=messages,
                bot=BotIdentity.MERRY,
                model_used=self._gemini_model,
            )

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            logger.warning("RESPONSE [merry]: Using fallback response")
            return ResponseResult(
                messages=["Tch... something's off. Let me think about this."],
                bot=BotIdentity.MERRY,
                model_used="fallback",
            )
