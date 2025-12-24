"""Memory extraction from conversations using Gemini Flash."""

import json
import logging
from dataclasses import dataclass

from google import genai
from google.genai import types

from bicker_bot.memory.store import Memory, MemoryStore, MemoryType

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
        model: str = "gemini-3-flash",
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
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=EXTRACTION_SYSTEM_PROMPT,
                    temperature=0.2,  # Low temperature for consistent extraction
                    max_output_tokens=1000,
                ),
            )

            raw = response.text.strip()

            # Parse JSON response
            # Handle potential markdown code blocks
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1])

            memories: list[Memory] = []

            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    for item in data:
                        memory = Memory(
                            content=item.get("content", ""),
                            user=item.get("user"),
                            memory_type=MemoryType(item.get("type", "fact")),
                            intensity=float(item.get("intensity", 0.5)),
                        )
                        if memory.content:  # Only add if there's content
                            memories.append(memory)

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse extraction response: {e}")

            # Store memories
            if memories:
                self._memory_store.add_batch(memories)
                logger.info(f"Extracted and stored {len(memories)} memories")

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
