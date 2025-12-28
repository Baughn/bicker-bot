"""ChromaDB-based memory store."""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field

from bicker_bot.config import MemoryConfig
from bicker_bot.core.logging import log_rag_query, log_rag_results
from bicker_bot.memory.embeddings import LocalEmbeddingFunction

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of memories."""

    FACT = "fact"  # Factual information about a user/topic
    OPINION = "opinion"  # User's opinions or preferences
    INTERACTION = "interaction"  # Notable interaction patterns
    EVENT = "event"  # Specific events that happened


class Memory(BaseModel):
    """A stored memory."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    user: str | None = None  # IRC nick, if user-specific
    memory_type: MemoryType = MemoryType.FACT
    intensity: float = Field(ge=0.0, le=1.0, default=0.5)
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_chroma_document(self) -> str:
        """Format memory for embedding."""
        parts = [self.content]
        if self.user:
            parts.insert(0, f"About {self.user}:")
        return " ".join(parts)

    def to_chroma_metadata(self) -> dict[str, Any]:
        """Convert to ChromaDB metadata format."""
        return {
            "user": self.user or "",
            "type": self.memory_type.value,
            "intensity": self.intensity,
            "timestamp": self.timestamp.isoformat(),
            **self.metadata,
        }


@dataclass
class SearchResult:
    """Result from a memory search."""

    memory: Memory
    distance: float  # Lower is more similar

    @property
    def similarity(self) -> float:
        """Convert distance to similarity score (0-1)."""
        # ChromaDB uses L2 distance by default
        # Convert to a rough similarity score
        return max(0.0, 1.0 - (self.distance / 2.0))


class MemoryStore:
    """ChromaDB-backed memory storage with local embeddings."""

    COLLECTION_NAME = "memories"

    def __init__(self, config: MemoryConfig):
        self._config = config
        self._embedding_fn = LocalEmbeddingFunction(config.embedding_model)

        # Ensure path exists
        chroma_path = Path(config.chroma_path)
        chroma_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB
        self._client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        logger.info(f"Memory store initialized at {chroma_path}")
        logger.info(f"Collection '{self.COLLECTION_NAME}' has {self._collection.count()} memories")

    def add(self, memory: Memory) -> str:
        """Add a memory to the store.

        Returns:
            The memory ID
        """
        self._collection.add(
            ids=[memory.id],
            documents=[memory.to_chroma_document()],
            metadatas=[memory.to_chroma_metadata()],
        )
        logger.info(
            f"MEMORY_ADD: id={memory.id[:8]}... user={memory.user or 'global'} "
            f"intensity={memory.intensity:.2f} type={memory.memory_type.value}"
        )
        return memory.id

    def add_batch(self, memories: list[Memory]) -> list[str]:
        """Add multiple memories at once."""
        if not memories:
            return []

        self._collection.add(
            ids=[m.id for m in memories],
            documents=[m.to_chroma_document() for m in memories],
            metadatas=[m.to_chroma_metadata() for m in memories],
        )
        intensities = [f"{m.intensity:.1f}" for m in memories]
        logger.info(f"MEMORY_BATCH_ADD: {len(memories)} memories (intensities: {intensities})")
        return [m.id for m in memories]

    def search(
        self,
        query: str,
        limit: int = 10,
        user: str | None = None,
        min_intensity: float | None = None,
        memory_types: list[MemoryType] | None = None,
    ) -> list[SearchResult]:
        """Search for relevant memories.

        Args:
            query: Search query
            limit: Maximum results to return
            user: Filter to specific user's memories
            min_intensity: Minimum intensity threshold
            memory_types: Filter to specific memory types

        Returns:
            List of search results, sorted by relevance
        """
        # Build where clause
        where_clauses: list[dict[str, Any]] = []

        if user:
            where_clauses.append({"user": {"$eq": user}})

        if min_intensity is not None:
            where_clauses.append({"intensity": {"$gte": min_intensity}})

        if memory_types:
            where_clauses.append({"type": {"$in": [t.value for t in memory_types]}})

        where: dict[str, Any] | None = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        # Log RAG query
        log_rag_query(
            operation="Memory Search",
            query=query,
            filters={
                "user": user,
                "min_intensity": min_intensity,
                "memory_types": [t.value for t in memory_types] if memory_types else None,
            },
            limit=limit,
        )

        # Query using the embedding function's query method
        query_embedding = self._embedding_fn.embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Parse results
        search_results: list[SearchResult] = []

        if results["ids"] and results["ids"][0]:
            for i, memory_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                document = results["documents"][0][i] if results["documents"] else ""
                distance = results["distances"][0][i] if results["distances"] else 0.0

                memory = Memory(
                    id=memory_id,
                    content=document,
                    user=metadata.get("user") or None,
                    memory_type=MemoryType(metadata.get("type", "fact")),
                    intensity=float(metadata.get("intensity", 0.5)),
                    timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                )

                search_results.append(SearchResult(memory=memory, distance=distance))

        # Log the search
        query_preview = query[:50] + "..." if len(query) > 50 else query
        logger.info(
            f"MEMORY_SEARCH: query='{query_preview}' user={user or 'any'} "
            f"limit={limit} -> {len(search_results)} results"
        )

        # Log top results at DEBUG
        for i, sr in enumerate(search_results[:3]):
            content_preview = sr.memory.content[:50] + "..." if len(sr.memory.content) > 50 else sr.memory.content
            logger.debug(f"  #{i+1}: sim={sr.similarity:.3f} '{content_preview}'")

        # Log RAG results for AI debug
        log_rag_results(
            operation="Memory Search",
            results=[sr.memory for sr in search_results],
            distances=[sr.distance for sr in search_results],
        )

        return search_results

    def get_user_memories(
        self,
        user: str,
        min_intensity: float | None = None,
        limit: int = 20,
    ) -> list[Memory]:
        """Get all memories for a specific user.

        Args:
            user: IRC nick
            min_intensity: Minimum intensity filter
            limit: Maximum results

        Returns:
            List of memories for the user
        """
        where: dict[str, Any] = {"user": {"$eq": user}}

        if min_intensity is not None:
            where = {"$and": [where, {"intensity": {"$gte": min_intensity}}]}

        results = self._collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )

        memories: list[Memory] = []
        if results["ids"]:
            for i, memory_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                document = results["documents"][i] if results["documents"] else ""

                memories.append(
                    Memory(
                        id=memory_id,
                        content=document,
                        user=metadata.get("user") or None,
                        memory_type=MemoryType(metadata.get("type", "fact")),
                        intensity=float(metadata.get("intensity", 0.5)),
                        timestamp=datetime.fromisoformat(
                            metadata.get("timestamp", datetime.now().isoformat())
                        ),
                    )
                )

        # Sort by intensity (highest first)
        memories.sort(key=lambda m: m.intensity, reverse=True)
        return memories

    def get_high_intensity_memories(self, user: str) -> list[Memory]:
        """Get high-intensity memories for a user.

        Uses the configured intensity threshold.
        """
        return self.get_user_memories(
            user=user,
            min_intensity=self._config.high_intensity_threshold,
        )

    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        try:
            self._collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    def count(self) -> int:
        """Get total number of memories."""
        return self._collection.count()

    def clear(self) -> None:
        """Delete all memories. Use with caution!"""
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Cleared all memories")

    def find_similar(
        self,
        content: str,
        threshold: float = 0.90,
        exclude_id: str | None = None,
    ) -> SearchResult | None:
        """Find the most similar existing memory above threshold.

        Args:
            content: Content to search for
            threshold: Minimum similarity threshold (0-1, cosine similarity)
            exclude_id: Optional memory ID to exclude from results

        Returns:
            Most similar memory above threshold, or None
        """
        if self._collection.count() == 0:
            return None

        # Query for single closest match
        query_embedding = self._embedding_fn.embed_query(content)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return None

        memory_id = results["ids"][0][0]

        # Skip if this is the excluded ID
        if exclude_id and memory_id == exclude_id:
            return None

        distance = results["distances"][0][0] if results["distances"] else 0.0

        # Convert cosine distance to similarity
        # ChromaDB with cosine space returns distance = 1 - similarity
        similarity = 1.0 - distance

        if similarity < threshold:
            return None

        metadata = results["metadatas"][0][0] if results["metadatas"] else {}
        document = results["documents"][0][0] if results["documents"] else ""

        memory = Memory(
            id=memory_id,
            content=document,
            user=metadata.get("user") or None,
            memory_type=MemoryType(metadata.get("type", "fact")),
            intensity=float(metadata.get("intensity", 0.5)),
            timestamp=datetime.fromisoformat(
                metadata.get("timestamp", datetime.now().isoformat())
            ),
        )

        return SearchResult(memory=memory, distance=distance)
