"""Memory extraction from conversations using Gemini Flash."""

import logging
from dataclasses import dataclass
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from bicker_bot.config import MemoryConfig
from bicker_bot.core.logging import get_session_stats, log_llm_call, log_llm_response, log_llm_round
from bicker_bot.memory.deduplicator import MemoryDeduplicator
from bicker_bot.memory.store import Memory, MemoryStore, MemoryType


class MemoryExtraction(BaseModel):
    """Schema for extracted memory from LLM."""

    content: str
    user: str | None = None
    type: Literal["fact", "opinion", "interaction", "event"] = "fact"
    intensity: float = Field(ge=0.0, le=1.0, default=0.5)


class MemoryExtractionResponse(BaseModel):
    """Wrapper for memory extractions (Gemini requires object at root)."""

    memories: list[MemoryExtraction] = Field(default_factory=list)

logger = logging.getLogger(__name__)


EXTRACTION_SYSTEM_PROMPT = """You are a memory extractor for an IRC chatbot system.
Your job is to identify memorable facts from conversations that should be stored for future reference.

Extract information that would be useful to remember about users, including:
- Personal information they share (interests, job, location, etc.)
- Opinions and preferences
- Important events they mention
- Promises or commitments made
- Recurring topics or jokes

For each memory, provide:
- content: The fact to remember. Include relevant context such as:
  - When it was mentioned (if notable)
  - Why it matters or how it came up
  - Related details that distinguish this from similar facts
  Aim for 1-3 sentences rather than fragments.
- user: The IRC nick this is about (if applicable)
- type: One of "fact", "opinion", "interaction", "event"
- intensity: How important is this?
  - 1.0: Explicit request to remember, or deeply personal
  - 0.8: Personal information, important preferences
  - 0.6: General opinions, casual preferences
  - 0.4: Topic interests, recurring patterns
  - 0.2: Minor observations, casual mentions

Respond with a JSON array of memory objects. If there's nothing worth remembering, return an empty array [].

Be selective - don't record every message, only genuinely memorable information.
Prefer richer, more contextual memories over short fragments.

Good: "Alice mentioned she adopted a rescue cat named Whiskers in 2023 after her previous cat passed away"
Bad: "Alice has a cat"
"""


@dataclass
class ExtractionResult:
    """Result of memory extraction."""

    memories_extracted: list[Memory]
    raw_response: str


class MemoryExtractor:
    """Extracts memories from conversations."""

    def __init__(
        self,
        api_key: str,
        memory_store: MemoryStore,
        model: str = "gemini-3-flash-preview",
        dedup_config: MemoryConfig | None = None,
    ):
        """Initialize the memory extractor.

        Args:
            api_key: Google AI API key
            memory_store: Where to store extracted memories
            model: Model to use for extraction
            dedup_config: Memory config with dedup settings (None disables dedup)
        """
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._memory_store = memory_store

        # Initialize deduplicator if enabled
        if dedup_config and dedup_config.dedup_enabled:
            self._deduplicator = MemoryDeduplicator(
                store=memory_store,
                api_key=api_key,
                model=model,
                upper_threshold=dedup_config.dedup_upper_threshold,
                lower_threshold=dedup_config.dedup_lower_threshold,
            )
        else:
            self._deduplicator = None

    async def extract_and_store(
        self,
        conversation: str,
        response_given: str,
        bot_name: str,
    ) -> ExtractionResult:
        """Extract memories from a conversation and store them.

        Args:
            conversation: Recent conversation context
            response_given: The response the bot just gave
            bot_name: Which bot responded

        Returns:
            ExtractionResult with extracted memories
        """
        prompt = f"""Recent conversation:
{conversation}

Bot ({bot_name}) responded: "{response_given}"

Extract any memorable facts from this conversation. Focus on what humans revealed about themselves.
Return a JSON array of memory objects, or [] if nothing is worth remembering."""

        try:
            # Log LLM input
            log_llm_call(
                operation="Memory Extraction",
                model=self._model,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_prompt=prompt,
                config={"temperature": 0.2, "max_output_tokens": 8192},
            )

            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    temperature=0.2,  # Low temperature for consistent extraction
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                    response_schema=types.Schema(
                        type=types.Type.OBJECT,
                        required=["memories"],
                        properties={
                            "memories": types.Schema(
                                type=types.Type.ARRAY,
                                items=types.Schema(
                                    type=types.Type.OBJECT,
                                    required=["content", "type", "intensity"],
                                    properties={
                                        "content": types.Schema(type=types.Type.STRING),
                                        "user": types.Schema(type=types.Type.STRING),
                                        "type": types.Schema(
                                            type=types.Type.STRING,
                                            enum=["fact", "opinion", "interaction", "event"],
                                        ),
                                        "intensity": types.Schema(type=types.Type.NUMBER),
                                    },
                                ),
                            ),
                        },
                    ),
                ),
            )

            # Log round summary (always on)
            usage = response.usage_metadata if response else None
            log_llm_round(
                component="extractor",
                model=self._model,
                round_num=1,
                tokens_in=usage.prompt_token_count if usage else None,
                tokens_out=usage.candidates_token_count if usage else None,
            )

            raw = response.text.strip()

            # Log LLM response
            log_llm_response(
                operation="Memory Extraction",
                response_text=raw,
            )

            # Parse and validate response with Pydantic
            memories: list[Memory] = []

            try:
                response_obj = MemoryExtractionResponse.model_validate_json(raw)
                for extraction in response_obj.memories:
                    memory = Memory(
                        content=extraction.content,
                        user=extraction.user,
                        memory_type=MemoryType(extraction.type),
                        intensity=extraction.intensity,
                    )
                    if memory.content:  # Only add if there's content
                        memories.append(memory)

            except Exception as e:
                # Shouldn't happen with structured output, but log if it does
                logger.error(f"Failed to parse memory extraction response: {e}\nRaw: {raw}")

            # Store memories (with deduplication if enabled)
            if memories:
                if self._deduplicator:
                    deduplicated = []
                    for memory in memories:
                        try:
                            result = await self._deduplicator.check_and_merge(memory)
                            deduplicated.append(result)
                        except Exception as e:
                            logger.warning(f"Dedup failed for memory, adding as-is: {e}")
                            deduplicated.append(memory)
                    self._memory_store.add_batch(deduplicated)
                    memories = deduplicated
                else:
                    self._memory_store.add_batch(memories)
                intensities = [f"{m.intensity:.1f}" for m in memories]
                logger.info(
                    f"MEMORY_EXTRACT: {len(memories)} memories "
                    f"(intensities: {intensities})"
                )
                # DEBUG log each memory
                for m in memories:
                    content_preview = m.content[:60] + "..." if len(m.content) > 60 else m.content
                    logger.debug(
                        f"  Extracted: [{m.memory_type.value}] "
                        f"{m.user or 'global'}: {content_preview}"
                    )

                # Track stats
                stats = get_session_stats()
                stats.increment("memories_stored", len(memories))
            else:
                logger.info("MEMORY_EXTRACT: 0 memories (nothing memorable)")

            # Track API call
            stats = get_session_stats()
            stats.increment_api_call(self._model)

            return ExtractionResult(
                memories_extracted=memories,
                raw_response=raw,
            )

        except Exception as e:
            logger.error(f"Memory extraction failed: {e}")
            return ExtractionResult(
                memories_extracted=[],
                raw_response=f"error: {e}",
            )

    async def extract_explicit_memory(
        self,
        user: str,
        content: str,
    ) -> Memory:
        """Store an explicitly requested memory.

        When a user says "remember that I..." this should be called directly.
        """
        memory = Memory(
            content=content,
            user=user,
            memory_type=MemoryType.FACT,
            intensity=1.0,  # Explicit requests get max intensity
        )
        self._memory_store.add(memory)
        logger.info(f"Stored explicit memory for {user}: {content[:50]}...")
        return memory
