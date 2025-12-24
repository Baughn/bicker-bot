"""Context builder using Gemini Flash with RAG tools."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types

from bicker_bot.core.logging import get_session_stats, log_llm_call, log_llm_response
from bicker_bot.memory import BotIdentity, Memory, MemoryStore

logger = logging.getLogger(__name__)


# Bot descriptions for context gathering
BOT_DESCRIPTIONS = {
    BotIdentity.MERRY: "a direct, action-oriented dream demon who dislikes overthinking and prefers to tackle problems head-on",
    BotIdentity.HACHIMAN: "a cynical observer who analyzes everything, uses self-deprecating humor, and secretly cares despite his pessimism",
}


def get_context_system_prompt(bot_nickname: str, bot_identity: BotIdentity) -> str:
    """Generate context system prompt with bot identity."""
    bot_description = BOT_DESCRIPTIONS.get(bot_identity, "an IRC chatbot")
    return f"""You are a context gatherer for {bot_nickname}, an IRC chatbot.

{bot_nickname} is {bot_description}.

Your job is to prepare context for {bot_nickname}'s response.

You have access to two tools:
1. rag_search - Search the memory database for relevant past information
2. ready_to_respond - Signal that you have enough context

Strategy:
1. Analyze the conversation and latest message
2. If you need information about users, topics, or past events, use rag_search
3. When you have sufficient context (or after 3 searches), call ready_to_respond with a summary

The summary should be concise bullet points of relevant context for the responding bot.
Do NOT generate the actual response - just gather context.

When calling ready_to_respond, provide a JSON summary with:
- key_facts: List of relevant facts discovered
- user_context: What we know about the user(s) involved
- topic_context: Relevant background on the topic
- suggested_tone: How the response should feel (playful, serious, helpful, etc.)
"""


# Tool definitions for Gemini
RAG_SEARCH_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
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
        ),
        types.FunctionDeclaration(
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
        ),
    ]
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
        model: str = "gemini-3-flash-preview",
    ):
        """Initialize the context builder.

        Args:
            api_key: Google AI API key
            memory_store: Memory store for RAG searches
            model: Model to use (default: gemini-3-flash)
        """
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._memory_store = memory_store

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

    async def build(
        self,
        message: str,
        recent_context: str,
        sender: str,
        high_intensity_memories: list[Memory],
        bot_identity: BotIdentity,
        bot_nickname: str,
    ) -> ContextResult:
        """Build context for a response.

        Args:
            message: The message to respond to
            recent_context: Recent conversation formatted
            sender: Who sent the message
            high_intensity_memories: Pre-fetched high-intensity memories for sender
            bot_identity: Which bot will respond (MERRY or HACHIMAN)
            bot_nickname: The IRC nickname of the responding bot

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

        initial_prompt = f"""Recent conversation:
{recent_context}

Latest message from {sender}: "{message}"
{hi_memories_str}
Analyze this and gather any additional context needed. Use rag_search if you need more information, or call ready_to_respond if you have enough."""

        # Generate system prompt with bot identity
        system_prompt = get_context_system_prompt(bot_nickname, bot_identity)

        # Log LLM input
        log_llm_call(
            operation="Context Build (initial)",
            model=self._model,
            system_prompt=system_prompt,
            user_prompt=initial_prompt,
            tools=[RAG_SEARCH_TOOL],
            config={"temperature": 0.3},
        )

        # Create chat session
        chat = self._client.aio.chats.create(
            model=self._model,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                tools=[RAG_SEARCH_TOOL],
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

            # Send tool results back
            if tool_results:
                # Log the tool results being sent
                log_llm_call(
                    operation=f"Context Build (round {result.rounds + 1} - tool results)",
                    model=self._model,
                    user_prompt=f"Tool results: {len(tool_results)} function responses",
                )
                response = await chat.send_message(tool_results)

        # If we exhausted rounds, force a summary
        if not result.summary:
            logger.warning(f"Context builder exhausted {self.MAX_ROUNDS} rounds without summary")
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

        return result
