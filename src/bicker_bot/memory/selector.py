"""Bot selector using personality embeddings."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from bicker_bot.config import MemoryConfig
from bicker_bot.memory.embeddings import LocalEmbeddingFunction

logger = logging.getLogger(__name__)


class BotIdentity(str, Enum):
    """Bot identities."""

    MERRY = "merry"
    HACHIMAN = "hachiman"


# Base personality descriptions for each bot
PERSONALITY_DESCRIPTIONS = {
    BotIdentity.MERRY: """
Merry Nightmare - Dream demon from Yumekui Merry.
Direct, action-oriented, jumps into things headfirst.
Gets frustrated when things don't make sense.
Abrasive on the surface but genuinely kind-hearted.
Prefers doing over thinking. Impatient with overthinking.
Topics: action, dreams, fighting, directness, courage, frustration, impatience,
getting things done, practical solutions, physical activities.
""",
    BotIdentity.HACHIMAN: """
Hachiman Hikigaya - Cynical loner from Oregairu.
Sharp wit, pessimistic observations, self-deprecating humor.
Overthinks everything, finds flaws in every plan.
Secretly caring despite the cynical exterior.
Prefers observation over action. Values authenticity.
Topics: cynicism, social dynamics, overthinking, observation, loneliness,
genuine connections, self-sacrifice, analysis, pessimism, literature.
""",
}


@dataclass
class SelectionResult:
    """Result of bot selection."""

    selected: BotIdentity
    merry_score: float
    hachiman_score: float
    reason: str

    @property
    def confidence(self) -> float:
        """How confident we are in the selection (0-1)."""
        total = self.merry_score + self.hachiman_score
        if total == 0:
            return 0.5
        winner_score = max(self.merry_score, self.hachiman_score)
        return winner_score / total


@dataclass
class BotSelector:
    """Selects which bot should respond based on message content."""

    config: MemoryConfig
    _embedding_fn: LocalEmbeddingFunction = field(init=False)
    _client: chromadb.PersistentClient = field(init=False)
    _collection: Any = field(init=False)
    _last_speaker: BotIdentity | None = field(default=None, init=False)
    _last_spoke: dict[BotIdentity, datetime] = field(default_factory=dict, init=False)
    _recent_messages: dict[BotIdentity, list[str]] = field(default_factory=dict, init=False)

    COLLECTION_NAME = "bot_personalities"
    MAX_RECENT_MESSAGES = 10

    def __post_init__(self) -> None:
        """Initialize the selector."""
        self._embedding_fn = LocalEmbeddingFunction(self.config.embedding_model)

        # Ensure path exists
        chroma_path = Path(self.config.chroma_path)
        chroma_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Initialize with base personalities if empty
        if self._collection.count() == 0:
            self._initialize_personalities()

        # Initialize tracking
        self._recent_messages = {
            BotIdentity.MERRY: [],
            BotIdentity.HACHIMAN: [],
        }

        logger.info("Bot selector initialized")

    def _initialize_personalities(self) -> None:
        """Initialize the collection with base personality embeddings."""
        logger.info("Initializing bot personality embeddings")

        ids = []
        documents = []
        metadatas = []

        for bot, description in PERSONALITY_DESCRIPTIONS.items():
            # Add base personality
            ids.append(f"{bot.value}_base")
            documents.append(description.strip())
            metadatas.append({"bot": bot.value, "type": "base"})

        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info(f"Added {len(ids)} personality embeddings")

    def _update_dynamic_embedding(self, bot: BotIdentity) -> None:
        """Update the dynamic embedding with recent messages."""
        recent = self._recent_messages.get(bot, [])
        if not recent:
            return

        # Combine personality with recent messages
        base_personality = PERSONALITY_DESCRIPTIONS[bot].strip()
        recent_text = " ".join(recent[-self.MAX_RECENT_MESSAGES :])
        combined = f"{base_personality}\n\nRecent messages:\n{recent_text}"

        doc_id = f"{bot.value}_dynamic"

        # Upsert the dynamic embedding
        existing = self._collection.get(ids=[doc_id])
        if existing["ids"]:
            self._collection.update(
                ids=[doc_id],
                documents=[combined],
                metadatas=[{"bot": bot.value, "type": "dynamic"}],
            )
        else:
            self._collection.add(
                ids=[doc_id],
                documents=[combined],
                metadatas=[{"bot": bot.value, "type": "dynamic"}],
            )

    def record_message(self, bot: BotIdentity, message: str) -> None:
        """Record that a bot sent a message.

        This updates the dynamic embeddings for more context-aware selection.
        """
        self._last_speaker = bot
        self._last_spoke[bot] = datetime.now()

        # Add to recent messages
        if bot not in self._recent_messages:
            self._recent_messages[bot] = []
        self._recent_messages[bot].append(message)

        # Trim to max size
        if len(self._recent_messages[bot]) > self.MAX_RECENT_MESSAGES:
            self._recent_messages[bot] = self._recent_messages[bot][-self.MAX_RECENT_MESSAGES :]

        # Update dynamic embedding
        self._update_dynamic_embedding(bot)

    def select(self, message: str) -> SelectionResult:
        """Select which bot should respond to a message.

        Uses embedding similarity to find which bot's personality
        is most relevant to the message content.
        """
        # Query for similar personality embeddings
        query_embedding = self._embedding_fn.embed_query(message)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=4,  # Get all embeddings (2 base + 2 dynamic max)
            include=["metadatas", "distances"],
        )

        # Calculate scores for each bot
        scores = {BotIdentity.MERRY: 0.0, BotIdentity.HACHIMAN: 0.0}

        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]

                bot = BotIdentity(metadata["bot"])
                doc_type = metadata.get("type", "base")

                # Convert distance to similarity (cosine distance, lower is better)
                similarity = 1.0 - distance

                # Weight dynamic embeddings slightly higher (recency bias)
                if doc_type == "dynamic":
                    similarity *= 1.2

                scores[bot] = max(scores[bot], similarity)

        # Apply recency penalty - whoever spoke last gets a slight penalty
        # to encourage alternation
        if self._last_speaker:
            scores[self._last_speaker] *= 0.85

        # Determine winner
        if scores[BotIdentity.MERRY] > scores[BotIdentity.HACHIMAN]:
            selected = BotIdentity.MERRY
            reason = "Message aligns more with Merry's direct, action-oriented style"
        elif scores[BotIdentity.HACHIMAN] > scores[BotIdentity.MERRY]:
            selected = BotIdentity.HACHIMAN
            reason = "Message aligns more with Hachiman's analytical, observational style"
        else:
            # Tie-breaker: whoever spoke less recently
            merry_last = self._last_spoke.get(BotIdentity.MERRY)
            hachi_last = self._last_spoke.get(BotIdentity.HACHIMAN)

            if merry_last is None:
                selected = BotIdentity.MERRY
            elif hachi_last is None:
                selected = BotIdentity.HACHIMAN
            elif merry_last < hachi_last:
                selected = BotIdentity.MERRY
            else:
                selected = BotIdentity.HACHIMAN
            reason = "Tie-breaker: choosing whoever spoke less recently"

        return SelectionResult(
            selected=selected,
            merry_score=scores[BotIdentity.MERRY],
            hachiman_score=scores[BotIdentity.HACHIMAN],
            reason=reason,
        )
