"""Response generator using Opus or Gemini Pro with web tools."""

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

import anthropic

from bicker_bot.core.logging import get_session_stats, log_llm_call, log_llm_response, log_llm_round
from bicker_bot.core.web import ImageData, WebFetcher, WebPageResult
from bicker_bot.memory.selector import BotIdentity
from bicker_bot.tracing import TraceContext


logger = logging.getLogger(__name__)


@dataclass
class ResponseResult:
    """Result of response generation."""

    messages: list[str] = field(default_factory=list)  # Can be empty, 1, or multiple
    bot: BotIdentity = BotIdentity.HACHIMAN
    model_used: str = ""
    truncated: bool = False  # True if output hit token limit (response should be discarded)


# Callback type for error notifications
ErrorNotifyCallback = Callable[[str], Coroutine[Any, Any, None]] | None


# Claude tool definition
CLAUDE_FETCH_TOOL = {
    "name": "fetch_webpage",
    "description": (
        "Fetch and read the content of a webpage. Returns the page content as markdown text, "
        "with any images included. Use this when someone shares a URL and you need to see "
        "what's on the page to respond helpfully. Generally avoid using this."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL to fetch (must start with http:// or https://)",
            },
            "include_images": {
                "type": "boolean",
                "description": "Whether to include images from the page (default: true)",
            },
        },
        "required": ["url"],
    },
}


# Fallback messages by bot personality
FALLBACK_MESSAGES = {
    BotIdentity.HACHIMAN: "...I had a thought, but it got away from me.",
    BotIdentity.MERRY: "Tch... something's off. Let me think about this.",
}


class ResponseGenerator:
    """Generates responses using Claude Opus."""

    MAX_TOOL_ROUNDS = 3

    def __init__(
        self,
        anthropic_api_key: str,
        google_api_key: str,  # Unused, kept for API consistency
        web_fetcher: WebFetcher | None = None,
        opus_model: str = "claude-opus-4-5-20251101",
        on_error_notify: ErrorNotifyCallback = None,
    ):
        """Initialize the response generator.

        Args:
            anthropic_api_key: Anthropic API key
            google_api_key: Unused, kept for API consistency with other components
            web_fetcher: WebFetcher instance for fetching URLs (optional)
            opus_model: Claude model for response generation
            on_error_notify: Callback to notify on critical errors (e.g., ping on IRC)
        """
        self._anthropic = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        self._web_fetcher = web_fetcher
        self._opus_model = opus_model
        self._on_error_notify = on_error_notify

    async def generate(
        self,
        bot: BotIdentity,
        system_prompt: str,
        context_summary: dict[str, Any],
        recent_conversation: str,
        message: str,
        sender: str,
        detected_urls: list[str] | None = None,
        trace_ctx: TraceContext | None = None,
    ) -> ResponseResult:
        """Generate a response as the specified bot.

        Args:
            bot: Which bot should respond
            system_prompt: The bot's personality prompt
            context_summary: Summary from context builder
            recent_conversation: Recent conversation formatted
            message: The message to respond to
            sender: Who sent the message
            detected_urls: URLs found in the message (optional, for convenience)

        Returns:
            ResponseResult with the generated content
        """
        # Format the context for the model
        context_str = self._format_context(context_summary)

        # Note about web capabilities
        web_note = ""
        if self._web_fetcher:
            web_note = """
You have access to a fetch_webpage tool. If the conversation includes a URL and you need
to see what's on that page to respond helpfully, use the tool to fetch it."""
            if detected_urls:
                web_note += f"\n\nURLs detected in message: {', '.join(detected_urls)}"

        user_prompt = f"""Recent conversation:
{recent_conversation}

Context gathered:
{context_str}

Latest message from {sender}: "{message}"
{web_note}
Respond naturally as your character. Keep it conversational and IRC-appropriate (not too long).
Do not use markdown formatting. Do not prefix your response with your name.

You can respond with:
- No messages (if nothing worth saying)
- One message (typical case)
- Multiple messages (if addressing multiple people or topics)
- Actions (IRC /me) by prefixing with "/me " - use sparingly for emphasis

Bias toward fewer messages unless specifically addressing multiple people.

Format your response as a raw JSON object, like so: {{"messages": ["/me yawns", "good morning"]}}

Do not include backticks. Do not nest the JSON.
"""

        return await self._generate_response(
            bot=bot,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            trace_ctx=trace_ctx,
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
        """Parse LLM response as JSON object with messages array."""
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            # Expected format: {"messages": ["msg1", "msg2"]}
            if isinstance(parsed, dict) and "messages" in parsed:
                messages = parsed["messages"]
                if isinstance(messages, list):
                    return [str(m) for m in messages if m]
            # Fallback: direct array (Claude format)
            if isinstance(parsed, list):
                return [str(m) for m in parsed if m]
            # Single string returned
            return [str(parsed)] if parsed else []
        except json.JSONDecodeError:
            # Fallback: treat raw text as single message
            return [raw.strip()] if raw.strip() else []

    def _format_webpage_result(self, result: WebPageResult) -> list[dict[str, Any]]:
        """Format a WebPageResult for Claude tool result.

        Returns a list of content blocks (text + images).
        """
        blocks: list[dict[str, Any]] = []

        # Add text content
        if result.error:
            text = f"Error fetching {result.url}: {result.error}"
        else:
            text = f"# {result.title}\n\n{result.markdown_content}"
            if result.truncated:
                text += "\n\n[Page content was truncated due to length]"

        blocks.append({"type": "text", "text": text})

        # Add images
        for img in result.images:
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.mime_type,
                    "data": img.base64_data,
                },
            })

        return blocks

    async def _generate_response(
        self,
        bot: BotIdentity,
        system_prompt: str,
        user_prompt: str,
        trace_ctx: TraceContext | None = None,
    ) -> ResponseResult:
        """Generate a response using Claude Opus with tool support."""
        bot_name = bot.value
        try:
            # Prepare tools if web fetcher is available
            tools = [CLAUDE_FETCH_TOOL] if self._web_fetcher else []

            # Log LLM input
            log_llm_call(
                operation=f"Response Generation ({bot_name})",
                model=self._opus_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tools if tools else None,
                config={"max_tokens": 8192},
            )

            messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]

            # Tool calling loop
            for round_num in range(self.MAX_TOOL_ROUNDS + 1):
                # On final round, omit tools to force a text response
                is_final_round = round_num >= self.MAX_TOOL_ROUNDS
                current_tools = anthropic.NOT_GIVEN if is_final_round else (tools if tools else anthropic.NOT_GIVEN)

                logger.debug(f"CLAUDE_ROUND {round_num}: sending {len(messages)} messages, tools={'omitted' if is_final_round else 'included'}")
                response = await self._anthropic.messages.create(
                    model=self._opus_model,
                    max_tokens=8192,
                    system=system_prompt,
                    tools=current_tools,
                    messages=messages,
                )
                logger.debug(
                    f"CLAUDE_ROUND {round_num}: stop_reason={response.stop_reason}, "
                    f"content_types={[b.type for b in response.content]}"
                )

                # Log round summary (always on)
                tool_names = [b.name for b in response.content if b.type == "tool_use"]
                log_llm_round(
                    component=f"responder/{bot_name}",
                    model=self._opus_model,
                    round_num=round_num,
                    tokens_in=response.usage.input_tokens,
                    tokens_out=response.usage.output_tokens,
                    tools_called=tool_names if tool_names else None,
                    stop_reason=response.stop_reason,
                )

                # Check for max_tokens stop reason - abort and discard
                if response.stop_reason == "max_tokens":
                    logger.error(
                        f"CLAUDE_MAX_TOKENS: Model {self._opus_model} hit token limit. "
                        f"Aborting response (will not be sent or stored in memory)."
                    )
                    if self._on_error_notify:
                        asyncio.create_task(
                            self._on_error_notify(
                                f"Baughn: Claude {self._opus_model} hit max_tokens! Response aborted."
                            )
                        )
                    # Add trace step for truncated response
                    if trace_ctx is not None:
                        trace_ctx.add_llm_step(
                            stage="responder",
                            inputs={
                                "bot": bot.value,
                                "message_preview": user_prompt[:200],
                            },
                            outputs={
                                "messages": [],
                                "truncated": True,
                            },
                            decision="TRUNCATED - max_tokens hit",
                            model=self._opus_model,
                            prompt=user_prompt,
                            raw_response="[truncated - max_tokens]",
                            thinking=None,
                            thought_signatures=None,
                            token_usage={
                                "input": response.usage.input_tokens,
                                "output": response.usage.output_tokens,
                            },
                        )
                    return ResponseResult(
                        messages=[],
                        bot=bot,
                        model_used=self._opus_model,
                        truncated=True,
                    )

                # Check for tool use
                tool_use_blocks = [
                    block for block in response.content
                    if block.type == "tool_use"
                ]

                if not tool_use_blocks or response.stop_reason != "tool_use":
                    # No tool calls, extract final response
                    break

                # Process tool calls
                messages.append({"role": "assistant", "content": response.content})

                tool_results: list[dict[str, Any]] = []
                for tool_block in tool_use_blocks:
                    if tool_block.name == "fetch_webpage" and self._web_fetcher:
                        url = tool_block.input.get("url", "")
                        include_images = tool_block.input.get("include_images", True)

                        logger.info(f"TOOL_CALL: fetch_webpage url={url}")

                        result = await self._web_fetcher.fetch(
                            url=url,
                            include_images=include_images,
                        )

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": self._format_webpage_result(result),
                        })

                messages.append({"role": "user", "content": tool_results})

            # Extract text response
            raw_content = None
            for block in response.content:
                if block.type == "text":
                    raw_content = block.text
                    break

            if raw_content is None:
                logger.warning(
                    f"CLAUDE_NO_TEXT: No text block in final response. "
                    f"Content types: {[b.type for b in response.content]}, "
                    f"stop_reason: {response.stop_reason}"
                )

            messages_out = self._parse_response_json(raw_content)

            # Log LLM response
            log_llm_response(
                operation=f"Response Generation ({bot_name})",
                response_text=raw_content,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )

            # Log response with token usage
            logger.info(
                f"RESPONSE [{bot_name}]: model={self._opus_model} "
                f"tokens_in={response.usage.input_tokens} "
                f"tokens_out={response.usage.output_tokens} "
                f"messages={len(messages_out)}"
            )
            logger.info(f"LITERAL_RESPONSE: {raw_content}")

            # Track stats
            stats = get_session_stats()
            stats.increment(f"responses_{bot_name}")
            stats.increment_api_call(self._opus_model)

            # Add trace step for successful response
            if trace_ctx is not None:
                # Extract thinking blocks if any
                thinking_text = None
                for block in response.content:
                    if block.type == "thinking":
                        thinking_text = block.thinking
                        break

                trace_ctx.add_llm_step(
                    stage="responder",
                    inputs={
                        "bot": bot.value,
                        "message_preview": user_prompt[:200],
                    },
                    outputs={
                        "messages": messages_out,
                        "truncated": False,
                    },
                    decision=f"{len(messages_out)} messages",
                    model=self._opus_model,
                    prompt=user_prompt,
                    raw_response=raw_content or "",
                    thinking=thinking_text,
                    thought_signatures=None,
                    token_usage={
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                    },
                )

            return ResponseResult(
                messages=messages_out,
                bot=bot,
                model_used=self._opus_model,
            )

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            logger.warning(f"RESPONSE [{bot_name}]: Using fallback response")

            # Add trace step for error/fallback case
            if trace_ctx is not None:
                trace_ctx.add_llm_step(
                    stage="responder",
                    inputs={
                        "bot": bot.value,
                        "message_preview": user_prompt[:200],
                    },
                    outputs={
                        "messages": [FALLBACK_MESSAGES[bot]],
                        "truncated": False,
                        "error": str(e),
                    },
                    decision="ERROR - using fallback",
                    model="fallback",
                    prompt=user_prompt,
                    raw_response=f"error: {e}",
                    thinking=None,
                    thought_signatures=None,
                    token_usage=None,
                )

            return ResponseResult(
                messages=[FALLBACK_MESSAGES[bot]],
                bot=bot,
                model_used="fallback",
            )

