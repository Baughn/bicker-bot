"""Context builder using Gemini Flash with RAG tools."""

import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types

from bicker_bot.core.logging import get_session_stats, log_llm_call, log_llm_response, log_llm_round
from bicker_bot.core.web import WebFetcher, WebPageResult
from bicker_bot.memory.selector import BotIdentity
from bicker_bot.memory.store import Memory, MemoryStore
from bicker_bot.tracing import TraceContext

logger = logging.getLogger(__name__)


# Bot descriptions for context gathering
BOT_DESCRIPTIONS = {
    BotIdentity.MERRY: "a direct, action-oriented dream demon who dislikes overthinking and prefers to tackle problems head-on",
    BotIdentity.HACHIMAN: "a cynical observer who analyzes everything, uses self-deprecating humor, and secretly cares despite his pessimism",
}


def get_context_system_prompt(
    bot_nickname: str, bot_identity: BotIdentity, has_web_fetcher: bool = False
) -> str:
    """Generate context system prompt with bot identity."""
    bot_description = BOT_DESCRIPTIONS.get(bot_identity, "an IRC chatbot")

    if has_web_fetcher:
        tools_section = """You have access to three tools:
1. rag_search - Search the memory database for relevant past information
2. fetch_webpage - Fetch and read a webpage (use when URLs are shared in the conversation)
3. ready_to_respond - Signal that you have enough context"""
    else:
        tools_section = """You have access to two tools:
1. rag_search - Search the memory database for relevant past information
2. ready_to_respond - Signal that you have enough context"""

    return f"""You are a context gatherer for {bot_nickname}, an IRC chatbot.

{bot_nickname} is {bot_description}.

Your job is to prepare context for {bot_nickname}'s response.

{tools_section}

Strategy:
1. Analyze the conversation and latest message
2. If you need information about users, topics, or past events, use rag_search
3. If someone shared a URL and understanding it would help, use fetch_webpage
4. When you have sufficient context (or after 3 tool uses), call ready_to_respond with a summary

The summary should be concise bullet points of relevant context for the responding bot.
Do NOT generate the actual response - just gather context.

When calling ready_to_respond, provide a JSON summary with:
- key_facts: List of relevant facts discovered
- user_context: What we know about the user(s) involved
- topic_context: Relevant background on the topic
- suggested_tone: How the response should feel (playful, serious, helpful, etc.)
"""


# Tool definitions for Gemini
RAG_SEARCH_DECLARATION = types.FunctionDeclaration(
    name="rag_search",
    description="Search the memory database for relevant past information about users, topics, or events",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(
                type=types.Type.STRING,
                description="Search query describing what information you need",
            ),
            "user": types.Schema(
                type=types.Type.STRING,
                description="Optional: filter to memories about a specific user",
            ),
        },
        required=["query"],
    ),
)

READY_TO_RESPOND_DECLARATION = types.FunctionDeclaration(
    name="ready_to_respond",
    description="Signal that you have gathered enough context and provide a summary",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "summary": types.Schema(
                type=types.Type.STRING,
                description="JSON summary of gathered context",
            ),
        },
        required=["summary"],
    ),
)

FETCH_WEBPAGE_DECLARATION = types.FunctionDeclaration(
    name="fetch_webpage",
    description=(
        "Fetch and read the content of a webpage. Returns the page content as markdown text, "
        "with any images included. Use this when someone shares a URL and you need to understand "
        "what's on the page to provide relevant context."
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

# Combined tool for early rounds (can search, fetch, or finish)
RAG_SEARCH_TOOL = types.Tool(
    function_declarations=[RAG_SEARCH_DECLARATION, READY_TO_RESPOND_DECLARATION]
)

# Tool set with web fetching enabled
RAG_AND_WEB_TOOL = types.Tool(
    function_declarations=[
        RAG_SEARCH_DECLARATION,
        FETCH_WEBPAGE_DECLARATION,
        READY_TO_RESPOND_DECLARATION,
    ]
)

# Final round tool - only ready_to_respond available
READY_ONLY_TOOL = types.Tool(
    function_declarations=[READY_TO_RESPOND_DECLARATION]
)


@dataclass
class ContextResult:
    """Result of context gathering."""

    summary: dict[str, Any] = field(default_factory=dict)
    search_queries: list[str] = field(default_factory=list)
    memories_found: list[Memory] = field(default_factory=list)
    rounds: int = 0


class ContextBuilder:
    """Builds context for responses using Gemini Flash with RAG."""

    MAX_ROUNDS = 3

    def __init__(
        self,
        api_key: str,
        memory_store: MemoryStore,
        web_fetcher: WebFetcher | None = None,
        model: str = "gemini-3-flash-preview",
    ):
        """Initialize the context builder.

        Args:
            api_key: Google AI API key
            memory_store: Memory store for RAG searches
            web_fetcher: WebFetcher for fetching URLs (optional)
            model: Model to use (default: gemini-3-flash)
        """
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._memory_store = memory_store
        self._web_fetcher = web_fetcher

    def _execute_rag_search(
        self,
        query: str,
        user: str | None = None,
    ) -> list[Memory]:
        """Execute a RAG search against the memory store."""
        results = self._memory_store.search(
            query=query,
            user=user,
            limit=5,
        )
        return [r.memory for r in results]

    def _format_search_results(self, memories: list[Memory]) -> str:
        """Format search results for the model."""
        if not memories:
            return "No relevant memories found."

        lines = []
        for m in memories:
            user_str = f"[about {m.user}] " if m.user else ""
            lines.append(f"- {user_str}{m.content}")
        return "\n".join(lines)

    def _format_webpage_result(self, result: WebPageResult) -> str:
        """Format a WebPageResult as text for the model."""
        if result.error:
            return f"Error fetching {result.url}: {result.error}"

        text = f"# {result.title}\n\n{result.markdown_content}"
        if result.truncated:
            text += "\n\n[Content truncated due to length]"
        if result.images:
            text += f"\n\n[{len(result.images)} images included]"
        return text

    async def build(
        self,
        message: str,
        recent_context: str,
        sender: str,
        high_intensity_memories: list[Memory],
        bot_identity: BotIdentity,
        bot_nickname: str,
        detected_urls: list[str] | None = None,
        trace_ctx: TraceContext | None = None,
    ) -> ContextResult:
        """Build context for a response.

        Args:
            message: The message to respond to
            recent_context: Recent conversation formatted
            sender: Who sent the message
            high_intensity_memories: Pre-fetched high-intensity memories for sender
            bot_identity: Which bot will respond (MERRY or HACHIMAN)
            bot_nickname: The IRC nickname of the responding bot
            detected_urls: URLs found in the message (optional)
            trace_ctx: Tracing context for debug observability (optional)

        Returns:
            ContextResult with gathered information
        """
        result = ContextResult()

        # Log start
        msg_preview = message[:50] + "..." if len(message) > 50 else message
        logger.info(f"CONTEXT_BUILD: Starting for sender={sender}, message='{msg_preview}'")

        # Format high-intensity memories
        hi_memories_str = ""
        if high_intensity_memories:
            hi_memories_str = "\n\nHigh-priority memories about sender:\n"
            for m in high_intensity_memories:
                hi_memories_str += f"- {m.content}\n"

        # Format detected URLs
        urls_str = ""
        if detected_urls and self._web_fetcher:
            urls_str = f"\n\nURLs detected in message: {', '.join(detected_urls)}"

        initial_prompt = f"""Recent conversation:
{recent_context}

Latest message from {sender}: "{message}"
{hi_memories_str}{urls_str}
Analyze this and gather any additional context needed. Use rag_search if you need more information, or call ready_to_respond if you have enough."""

        # Choose tool set based on web fetcher availability
        has_web_fetcher = self._web_fetcher is not None
        tools = [RAG_AND_WEB_TOOL] if has_web_fetcher else [RAG_SEARCH_TOOL]

        # Generate system prompt with bot identity
        system_prompt = get_context_system_prompt(bot_nickname, bot_identity, has_web_fetcher)

        # Log LLM input
        log_llm_call(
            operation="Context Build (initial)",
            model=self._model,
            system_prompt=system_prompt,
            user_prompt=initial_prompt,
            tools=tools,
            config={"temperature": 0.3},
        )

        # Create chat session
        chat = self._client.aio.chats.create(
            model=self._model,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                max_output_tokens=8192,
                tools=tools,
                thinking_config=types.ThinkingConfig(
                    thinkingLevel=types.ThinkingLevel.LOW,
                ),
            ),
        )

        # Initial message
        response = await chat.send_message(initial_prompt)

        # Tool use loop
        while result.rounds < self.MAX_ROUNDS:
            result.rounds += 1

            # Check for tool calls
            if not response.candidates or not response.candidates[0].content.parts:
                break

            tool_calls = []
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    tool_calls.append(part.function_call)

            # Log the response from this round
            response_text = None
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    response_text = part.text
                    break

            log_llm_response(
                operation=f"Context Build (round {result.rounds})",
                response_text=response_text,
                tool_calls=[{"name": tc.name, "args": dict(tc.args)} for tc in tool_calls] if tool_calls else None,
            )

            # Log round summary (always on)
            tool_names = [tc.name for tc in tool_calls] if tool_calls else None
            usage = response.usage_metadata if response else None
            log_llm_round(
                component="context",
                model=self._model,
                round_num=result.rounds,
                tokens_in=usage.prompt_token_count if usage else None,
                tokens_out=usage.candidates_token_count if usage else None,
                tools_called=tool_names,
            )

            if not tool_calls:
                # No tool calls, might have a text response
                break

            # Process tool calls
            tool_results = []
            for call in tool_calls:
                if call.name == "ready_to_respond":
                    # Extract summary and finish
                    try:
                        summary_str = call.args.get("summary", "{}")
                        result.summary = json.loads(summary_str)
                    except json.JSONDecodeError:
                        result.summary = {"raw": summary_str}

                    # Log completion
                    logger.info(
                        f"CONTEXT_BUILD: Complete in {result.rounds} rounds, "
                        f"{len(result.search_queries)} searches, "
                        f"{len(result.memories_found)} memories"
                    )

                    # Track API call
                    stats = get_session_stats()
                    stats.increment_api_call(self._model)

                    # Add trace step if context provided
                    if trace_ctx is not None:
                        trace_ctx.add_llm_step(
                            stage="context",
                            inputs={
                                "message": message[:200],
                                "sender": sender,
                                "high_intensity_count": len(high_intensity_memories),
                            },
                            outputs={
                                "summary": result.summary,
                                "rounds": result.rounds,
                                "memories_found": len(result.memories_found),
                            },
                            decision=f"{result.rounds} rounds, {len(result.memories_found)} memories",
                            model=self._model,
                            prompt=initial_prompt,
                            raw_response=str(result.summary),
                            thinking=None,
                            thought_signatures=None,
                            token_usage=None,  # Aggregate across rounds not easily available
                            details={
                                "search_queries": result.search_queries,
                            },
                        )

                    return result

                elif call.name == "rag_search":
                    query = call.args.get("query", "")
                    user = call.args.get("user")
                    result.search_queries.append(query)

                    memories = self._execute_rag_search(query, user)
                    result.memories_found.extend(memories)

                    # Log the RAG search
                    logger.info(
                        f"RAG_SEARCH: query='{query}' user={user or 'any'} -> "
                        f"{len(memories)} memories found"
                    )

                    # Track stats
                    stats = get_session_stats()
                    stats.increment("memories_searched")
                    stats.increment("memories_found", len(memories))

                    tool_results.append(
                        types.Part.from_function_response(
                            name="rag_search",
                            response={"result": self._format_search_results(memories)},
                        )
                    )

                elif call.name == "fetch_webpage" and self._web_fetcher:
                    url = call.args.get("url", "")
                    include_images = call.args.get("include_images", True)

                    logger.info(f"CONTEXT_FETCH: url={url}")

                    webpage_result = await self._web_fetcher.fetch(
                        url=url,
                        include_images=include_images,
                    )

                    # Format as text + images
                    parts: list[types.Part] = [
                        types.Part.from_function_response(
                            name="fetch_webpage",
                            response={"content": self._format_webpage_result(webpage_result)},
                        )
                    ]

                    # Add images as inline data
                    for img in webpage_result.images:
                        try:
                            image_bytes = base64.b64decode(img.base64_data)
                            parts.append(
                                types.Part.from_bytes(
                                    data=image_bytes,
                                    mime_type=img.mime_type,
                                )
                            )
                        except Exception as img_err:
                            logger.debug(f"Failed to add image to context: {img_err}")

                    tool_results.extend(parts)

                    # Track stats
                    stats = get_session_stats()
                    stats.increment("webpages_fetched")

            # Send tool results back
            if tool_results:
                # Log the tool results being sent
                log_llm_call(
                    operation=f"Context Build (round {result.rounds + 1} - tool results)",
                    model=self._model,
                    user_prompt=f"Tool results: {len(tool_results)} function responses",
                )
                response = await chat.send_message(tool_results)

        # If we exhausted rounds, force a final summary call
        if not result.summary:
            logger.info(f"Context builder exhausted {self.MAX_ROUNDS} rounds, forcing summary")

            # Make one final call with only ready_to_respond available
            final_prompt = (
                "You have completed your searches. You MUST now call ready_to_respond "
                "with a summary of what you found. This is your only available action."
            )

            try:
                final_response = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=[
                        {"role": "user", "parts": [{"text": initial_prompt}]},
                        {"role": "model", "parts": [{"text": "I'll gather context now."}]},
                        {"role": "user", "parts": [{"text": final_prompt}]},
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.3,
                        tools=[READY_ONLY_TOOL],
                    ),
                )

                # Check for ready_to_respond call
                if final_response.candidates and final_response.candidates[0].content.parts:
                    for part in final_response.candidates[0].content.parts:
                        if part.function_call and part.function_call.name == "ready_to_respond":
                            try:
                                summary_str = part.function_call.args.get("summary", "{}")
                                result.summary = json.loads(summary_str)
                            except json.JSONDecodeError:
                                result.summary = {"raw": summary_str}
                            break

            except Exception as e:
                logger.warning(f"Final summary call failed: {e}")

            # If still no summary, use fallback
            if not result.summary:
                logger.warning("Context builder could not get summary, using fallback")
                result.summary = {
                    "key_facts": [],
                    "user_context": "Limited information available",
                    "topic_context": "General conversation",
                    "suggested_tone": "friendly",
                }

        # Log completion (for fallback path)
        logger.info(
            f"CONTEXT_BUILD: Complete in {result.rounds} rounds, "
            f"{len(result.search_queries)} searches, "
            f"{len(result.memories_found)} memories"
        )

        # Track API call
        stats = get_session_stats()
        stats.increment_api_call(self._model)

        # Add trace step if context provided
        if trace_ctx is not None:
            trace_ctx.add_llm_step(
                stage="context",
                inputs={
                    "message": message[:200],
                    "sender": sender,
                    "high_intensity_count": len(high_intensity_memories),
                },
                outputs={
                    "summary": result.summary,
                    "rounds": result.rounds,
                    "memories_found": len(result.memories_found),
                },
                decision=f"{result.rounds} rounds, {len(result.memories_found)} memories",
                model=self._model,
                prompt=initial_prompt,
                raw_response=str(result.summary),
                thinking=None,
                thought_signatures=None,
                token_usage=None,  # Aggregate across rounds not easily available
                details={
                    "search_queries": result.search_queries,
                },
            )

        return result
