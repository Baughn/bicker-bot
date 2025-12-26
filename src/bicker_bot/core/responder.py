"""Response generator using Opus or Gemini Pro with web tools."""

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

import anthropic
from google import genai
from google.genai import types

from bicker_bot.core.logging import get_session_stats, log_llm_call, log_llm_response, log_llm_round
from bicker_bot.core.web import ImageData, WebFetcher, WebPageResult
from bicker_bot.memory import BotIdentity


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
        "what's on the page to respond helpfully."
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


# Gemini tool definition
GEMINI_FETCH_DECLARATION = types.FunctionDeclaration(
    name="fetch_webpage",
    description=(
        "Fetch and read the content of a webpage. Returns the page content as markdown text, "
        "with any images included. Use this when someone shares a URL and you need to see "
        "what's on the page to respond helpfully."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "url": types.Schema(
                type=types.Type.STRING,
                description="The full URL to fetch (must start with http:// or https://)",
            ),
            "include_images": types.Schema(
                type=types.Type.BOOLEAN,
                description="Whether to include images from the page (default: true)",
            ),
        },
        required=["url"],
    ),
)

GEMINI_FETCH_TOOL = types.Tool(function_declarations=[GEMINI_FETCH_DECLARATION])


class ResponseGenerator:
    """Generates responses using either Claude Opus or Gemini Pro."""

    MAX_TOOL_ROUNDS = 3

    def __init__(
        self,
        anthropic_api_key: str,
        google_api_key: str,
        web_fetcher: WebFetcher | None = None,
        opus_model: str = "claude-opus-4-5-20251101",
        gemini_model: str = "gemini-3-pro-preview",
        on_error_notify: ErrorNotifyCallback = None,
    ):
        """Initialize the response generator.

        Args:
            anthropic_api_key: Anthropic API key
            google_api_key: Google AI API key
            web_fetcher: WebFetcher instance for fetching URLs (optional)
            opus_model: Claude model for Hachiman
            gemini_model: Gemini model for Merry
            on_error_notify: Callback to notify on critical errors (e.g., ping on IRC)
        """
        self._anthropic = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        self._gemini = genai.Client(api_key=google_api_key)
        self._web_fetcher = web_fetcher
        self._opus_model = opus_model
        self._gemini_model = gemini_model
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

Format your response as a raw JSON array of strings, like:
["first message"]
or for multiple: ["first message", "second message"]
or with an action: ["/me sighs", "Fine, I'll help."]
or for no response: []

Do not include a "json```" header or ``` trailer."""

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

    def _format_webpage_result_text(self, result: WebPageResult) -> str:
        """Format a WebPageResult as text only (for Gemini)."""
        if result.error:
            return f"Error fetching {result.url}: {result.error}"

        text = f"# {result.title}\n\n{result.markdown_content}"
        if result.truncated:
            text += "\n\n[Page content was truncated due to length]"

        if result.images:
            text += f"\n\n[Page contains {len(result.images)} images]"

        return text

    async def _generate_opus(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> ResponseResult:
        """Generate a response using Claude Opus with tool support."""
        try:
            # Prepare tools if web fetcher is available
            tools = [CLAUDE_FETCH_TOOL] if self._web_fetcher else []

            # Log LLM input
            log_llm_call(
                operation="Response Generation (Hachiman)",
                model=self._opus_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tools if tools else None,
                config={"max_tokens": 4096},
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
                    max_tokens=4096,
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
                    component="responder/hachiman",
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
                    return ResponseResult(
                        messages=[],
                        bot=BotIdentity.HACHIMAN,
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
                f"messages={len(messages_out)}"
            )
            logger.info(f"LITERAL_RESPONSE: {raw_content}")

            # Track stats
            stats = get_session_stats()
            stats.increment("responses_hachiman")
            stats.increment_api_call(self._opus_model)

            return ResponseResult(
                messages=messages_out,
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
        """Generate a response using Gemini Pro with tool support."""
        try:
            # Prepare tools and config
            tools = [GEMINI_FETCH_TOOL] if self._web_fetcher else None

            # Log LLM input
            log_llm_call(
                operation="Response Generation (Merry)",
                model=self._gemini_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tools,
                config={"temperature": 0.8, "max_output_tokens": 4096, "thinking": "LOW"},
            )

            if tools:
                # Use chat session for tool calling
                chat = self._gemini.aio.chats.create(
                    model=self._gemini_model,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.8,
                        max_output_tokens=4096,
                        tools=tools,
                        thinking_config=types.ThinkingConfig(
                            thinkingLevel=types.ThinkingLevel.LOW,
                        ),
                    ),
                )

                response = await chat.send_message(user_prompt)

                # Tool calling loop
                for round_num in range(self.MAX_TOOL_ROUNDS):
                    # Log response info
                    part_types = []
                    if response.candidates and response.candidates[0].content.parts:
                        part_types = [
                            "function_call" if p.function_call else
                            "function_response" if p.function_response else
                            "text" if p.text else "other"
                            for p in response.candidates[0].content.parts
                        ]
                    logger.debug(
                        f"GEMINI_ROUND {round_num}: part_types={part_types}, "
                        f"finish_reason={response.candidates[0].finish_reason if response.candidates else None}"
                    )

                    # Log round summary (always on)
                    tool_names = [p.function_call.name for p in (response.candidates[0].content.parts if response.candidates and response.candidates[0].content.parts else []) if p.function_call]
                    usage = response.usage_metadata if response else None
                    log_llm_round(
                        component="responder/merry",
                        model=self._gemini_model,
                        round_num=round_num,
                        tokens_in=usage.prompt_token_count if usage else None,
                        tokens_out=usage.candidates_token_count if usage else None,
                        tools_called=tool_names if tool_names else None,
                    )

                    # Check for function calls
                    function_calls = []
                    if response.candidates and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if part.function_call:
                                function_calls.append(part.function_call)

                    if not function_calls:
                        break

                    # On final round, ask for text response instead of processing more tools
                    if round_num >= self.MAX_TOOL_ROUNDS - 1:
                        logger.debug("GEMINI: Final round, requesting text response")
                        response = await chat.send_message(
                            "Please provide your final response now based on the information gathered. "
                            "Do not make any more tool calls."
                        )
                        break

                    # Process function calls
                    tool_results = []
                    for call in function_calls:
                        if call.name == "fetch_webpage" and self._web_fetcher:
                            url = call.args.get("url", "")
                            include_images = call.args.get("include_images", True)

                            logger.info(f"TOOL_CALL: fetch_webpage url={url}")

                            result = await self._web_fetcher.fetch(
                                url=url,
                                include_images=include_images,
                            )

                            # For Gemini, include images as separate parts
                            parts = [
                                types.Part.from_function_response(
                                    name="fetch_webpage",
                                    response={"content": self._format_webpage_result_text(result)},
                                )
                            ]

                            # Add images as inline data
                            for img in result.images:
                                try:
                                    import base64
                                    image_bytes = base64.b64decode(img.base64_data)
                                    parts.append(
                                        types.Part.from_bytes(
                                            data=image_bytes,
                                            mime_type=img.mime_type,
                                        )
                                    )
                                except Exception as img_err:
                                    logger.debug(f"Failed to add image to Gemini: {img_err}")

                            tool_results.extend(parts)

                    if tool_results:
                        response = await chat.send_message(tool_results)
            else:
                # No tools, simple generation
                response = await self._gemini.aio.models.generate_content(
                    model=self._gemini_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.8,
                        max_output_tokens=4096,
                        response_mime_type="application/json",
                        response_schema=types.Schema(
                            type=types.Type.OBJECT,
                            required=["messages"],
                            properties={
                                "messages": types.Schema(
                                    type=types.Type.ARRAY,
                                    items=types.Schema(
                                        type=types.Type.STRING,
                                    ),
                                ),
                            },
                        ),
                        thinking_config=types.ThinkingConfig(
                            thinkingLevel=types.ThinkingLevel.LOW,
                        ),
                    ),
                )

            # Check for MAX_TOKENS finish reason - abort and discard
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
                if finish_reason == types.FinishReason.MAX_TOKENS:
                    logger.error(
                        f"GEMINI_MAX_TOKENS: Model {self._gemini_model} hit token limit. "
                        f"Aborting response (will not be sent or stored in memory)."
                    )
                    if self._on_error_notify:
                        asyncio.create_task(
                            self._on_error_notify(
                                f"Baughn: Gemini {self._gemini_model} hit max_tokens! Response aborted."
                            )
                        )
                    return ResponseResult(
                        messages=[],
                        bot=BotIdentity.MERRY,
                        model_used=self._gemini_model,
                        truncated=True,
                    )

            raw_content = response.text if response else None
            messages = self._parse_response_json(raw_content)

            # Log LLM response
            log_llm_response(
                operation="Response Generation (Merry)",
                response_text=raw_content,
            )

            # Log response
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
