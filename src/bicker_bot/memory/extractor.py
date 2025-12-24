"""Memory extraction from conversations using Gemini Flash."""

import logging
from dataclasses import dataclass
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, RootModel

from bicker_bot.core.logging import get_session_stats, log_llm_call, log_llm_response
from bicker_bot.memory.store import Memory, MemoryStore, MemoryType


class MemoryExtraction(BaseModel):
    """Schema for extracted memory from LLM."""

    content: str
    user: str | None = None
    type: Literal["fact", "opinion", "interaction", "event"] = "fact"
    intensity: float = Field(ge=0.0, le=1.0, default=0.5)


class MemoryExtractionList(RootModel[list[MemoryExtraction]]):
    """List of memory extractions for JSON schema."""

    pass

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
- content: The fact to remember (concise, clear)
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
    ):
        """Initialize the memory extractor.

        Args:
            api_key: Google AI API key
            memory_store: Where to store extracted memories
            model: Model to use for extraction
        """
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._memory_store = memory_store

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
                config={"temperature": 0.2, "max_output_tokens": 1000},
            )

            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    temperature=0.2,  # Low temperature for consistent extraction
                    max_output_tokens=1000,
                    response_mime_type="application/json",
                    response_json_schema=MemoryExtractionList.model_json_schema(),
                ),
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
                extractions = MemoryExtractionList.model_validate_json(raw)
                for extraction in extractions.root:
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

            # Store memories
            if memories:
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
