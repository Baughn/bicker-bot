"""Memory deduplication for write-time and batch cleanup."""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime

from google import genai
from google.genai import types

from bicker_bot.memory.store import Memory, MemoryStore

logger = logging.getLogger(__name__)


MERGE_PROMPT = """You are merging two similar memories about the same topic.
Synthesize them into a single, comprehensive memory.

Rules:
- Combine all distinct information from both
- If there's conflicting information, prefer the NEWER memory
- Keep the result concise but complete
- Preserve the user attribution if present

OLDER memory (recorded {timestamp_a}):
{content_a}

NEWER memory (recorded {timestamp_b}):
{content_b}

Return only the merged memory text, nothing else."""


CLUSTER_MERGE_PROMPT = """You are merging {count} similar memories about the same topic.
Synthesize them into a single, comprehensive memory.

Rules:
- Combine all distinct information from all memories
- If there's conflicting information, prefer the NEWER memories
- Keep the result concise but complete
- Preserve the user attribution if present

Memories (oldest to newest):
{memories}

Return only the merged memory text, nothing else."""


@dataclass
class DeduplicationReport:
    """Report from batch deduplication."""

    memories_scanned: int = 0
    clusters_found: int = 0
    memories_merged: int = 0
    memories_deleted: int = 0
    merge_details: list[dict] = field(default_factory=list)


class MemoryDeduplicator:
    """Handles memory deduplication at write-time and in batch mode.

    Two-threshold logic:
    - similarity >= upper_threshold: auto-replace (delete old, keep new)
    - similarity in [lower_threshold, upper_threshold): LLM merges the two
    - similarity < lower_threshold: add as new memory
    """

    def __init__(
        self,
        store: MemoryStore,
        api_key: str,
        model: str = "gemini-2.0-flash",
        upper_threshold: float = 0.95,
        lower_threshold: float = 0.90,
    ):
        """Initialize the deduplicator.

        Args:
            store: Memory store to operate on
            api_key: Google API key for LLM merging
            model: Model to use for merging (default: gemini-2.0-flash)
            upper_threshold: Similarity above which to auto-replace
            lower_threshold: Similarity above which to trigger LLM merge
        """
        self._store = store
        self._model = model
        self._upper_threshold = upper_threshold
        self._lower_threshold = lower_threshold

        # Initialize Gemini client
        self._client = genai.Client(api_key=api_key)

    async def check_and_merge(self, new_memory: Memory) -> Memory:
        """Check a new memory against existing and merge if needed.

        This is the write-time deduplication entry point.

        Args:
            new_memory: New memory to check

        Returns:
            The memory to store (may be merged or unchanged)
        """
        # Find the most similar existing memory
        similar = self._store.find_similar(
            content=new_memory.content,
            threshold=self._lower_threshold,
        )

        if similar is None:
            # No similar memory found, use as-is
            logger.debug(
                f"DEDUP_CHECK: '{new_memory.content[:50]}...' no similar match -> add as new"
            )
            return new_memory

        # Calculate similarity from distance (cosine space: similarity = 1 - distance)
        similarity = 1.0 - similar.distance

        logger.debug(
            f"DEDUP_CHECK: '{new_memory.content[:50]}...' "
            f"closest match sim={similarity:.3f}"
        )

        if similarity >= self._upper_threshold:
            # Auto-replace: delete old, return new unchanged
            logger.info(
                f"DEDUP_AUTO_REPLACE: sim={similarity:.3f} >= {self._upper_threshold} "
                f"-> deleting old memory {similar.memory.id[:8]}..."
            )
            self._store.delete(similar.memory.id)
            return new_memory

        # Gray zone: LLM merge
        logger.info(
            f"DEDUP_MERGE: sim={similarity:.3f} in "
            f"[{self._lower_threshold}, {self._upper_threshold}) "
            f"-> LLM merging with {similar.memory.id[:8]}..."
        )

        try:
            merged_content = await self._merge_with_llm(similar.memory, new_memory)

            if not merged_content or not merged_content.strip():
                logger.warning("DEDUP_FAIL: LLM returned empty merge, falling back to replace")
                self._store.delete(similar.memory.id)
                return new_memory

            # Create merged memory with combined metadata
            merged_memory = Memory(
                content=merged_content,
                user=new_memory.user or similar.memory.user,
                memory_type=new_memory.memory_type,
                intensity=max(new_memory.intensity, similar.memory.intensity),
                timestamp=new_memory.timestamp,  # Use newer timestamp
                metadata={**similar.memory.metadata, **new_memory.metadata},
            )

            # Delete old memory
            self._store.delete(similar.memory.id)

            logger.info(
                f"DEDUP_MERGE: 2 memories -> '{merged_content[:50]}...'"
            )
            return merged_memory

        except Exception as e:
            logger.warning(f"DEDUP_FAIL: LLM merge failed ({e}), falling back to replace")
            self._store.delete(similar.memory.id)
            return new_memory

    async def _merge_with_llm(self, old_memory: Memory, new_memory: Memory) -> str:
        """Use LLM to merge two similar memories.

        Args:
            old_memory: The older existing memory
            new_memory: The newer memory being added

        Returns:
            Merged content string
        """
        prompt = MERGE_PROMPT.format(
            timestamp_a=old_memory.timestamp.isoformat(),
            content_a=old_memory.content,
            timestamp_b=new_memory.timestamp.isoformat(),
            content_b=new_memory.content,
        )

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,  # Low temperature for consistent merging
                max_output_tokens=500,
            ),
        )

        return response.text.strip() if response.text else ""

    async def _merge_cluster(self, memories: list[Memory]) -> str:
        """Use LLM to merge a cluster of similar memories.

        Args:
            memories: List of memories to merge (sorted oldest to newest)

        Returns:
            Merged content string
        """
        if len(memories) == 2:
            return await self._merge_with_llm(memories[0], memories[1])

        # Format memories for the prompt
        memories_text = "\n\n".join(
            f"[{i+1}] ({m.timestamp.isoformat()}):\n{m.content}"
            for i, m in enumerate(memories)
        )

        prompt = CLUSTER_MERGE_PROMPT.format(
            count=len(memories),
            memories=memories_text,
        )

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=500,
            ),
        )

        return response.text.strip() if response.text else ""

    async def deduplicate_all(self, dry_run: bool = False) -> DeduplicationReport:
        """Run batch deduplication on all memories.

        Uses union-find to cluster similar memories, then merges each cluster.

        Args:
            dry_run: If True, report what would be done without modifying

        Returns:
            DeduplicationReport with statistics
        """
        report = DeduplicationReport()

        # Get all memories
        all_memories = self._get_all_memories()
        report.memories_scanned = len(all_memories)

        if not all_memories:
            return report

        # Build similarity clusters using union-find
        clusters = self._build_clusters(all_memories)
        report.clusters_found = len([c for c in clusters if len(c) > 1])

        # Process each cluster with more than one memory
        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            # Sort by timestamp (oldest first)
            cluster.sort(key=lambda m: m.timestamp)

            if dry_run:
                report.merge_details.append({
                    "action": "would_merge",
                    "count": len(cluster),
                    "contents": [m.content[:50] for m in cluster],
                })
                report.memories_merged += 1
                report.memories_deleted += len(cluster) - 1
            else:
                try:
                    merged_content = await self._merge_cluster(cluster)

                    if not merged_content or not merged_content.strip():
                        logger.warning(
                            f"BATCH_DEDUP: Empty merge for cluster of {len(cluster)}, skipping"
                        )
                        continue

                    # Create merged memory from newest
                    newest = cluster[-1]
                    merged_memory = Memory(
                        content=merged_content,
                        user=newest.user,
                        memory_type=newest.memory_type,
                        intensity=max(m.intensity for m in cluster),
                        timestamp=newest.timestamp,
                    )

                    # Delete all old memories
                    for memory in cluster:
                        self._store.delete(memory.id)

                    # Add merged memory
                    self._store.add(merged_memory)

                    report.memories_merged += 1
                    report.memories_deleted += len(cluster) - 1
                    report.merge_details.append({
                        "action": "merged",
                        "count": len(cluster),
                        "result": merged_content[:50],
                    })

                    logger.info(
                        f"BATCH_DEDUP: Merged {len(cluster)} memories -> '{merged_content[:50]}...'"
                    )

                except Exception as e:
                    logger.error(f"BATCH_DEDUP: Failed to merge cluster: {e}")

        return report

    def _get_all_memories(self) -> list[Memory]:
        """Get all memories from the store.

        Returns:
            List of all memories
        """
        # Use a broad search to get all memories
        # This is a workaround since ChromaDB doesn't have a direct "get all" with embeddings
        count = self._store.count()
        if count == 0:
            return []

        # Get memories using the collection directly
        results = self._store._collection.get(
            limit=count,
            include=["documents", "metadatas"],
        )

        memories = []
        if results["ids"]:
            from bicker_bot.memory.store import MemoryType

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

        return memories

    def _build_clusters(self, memories: list[Memory]) -> list[list[Memory]]:
        """Build clusters of similar memories using union-find.

        Uses pairwise similarity comparison to find all similar pairs,
        then groups them using union-find for transitive clustering.

        Args:
            memories: List of all memories

        Returns:
            List of clusters (each cluster is a list of similar memories)
        """
        n = len(memories)
        if n == 0:
            return []

        # Union-find data structure
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px == py:
                return
            # Union by rank
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Get embeddings for all memories to compute pairwise similarity
        # This is more efficient than querying ChromaDB n^2 times
        embedding_fn = self._store._embedding_fn
        documents = [m.to_chroma_document() for m in memories]
        embeddings = embedding_fn.embed_documents(documents)

        # Compute pairwise cosine similarities and union similar pairs
        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        for i in range(n):
            for j in range(i + 1, n):
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                if similarity >= self._lower_threshold:
                    union(i, j)

        # Build clusters from union-find
        cluster_map: dict[int, list[Memory]] = {}
        for i in range(n):
            root = find(i)
            if root not in cluster_map:
                cluster_map[root] = []
            cluster_map[root].append(memories[i])

        return list(cluster_map.values())
