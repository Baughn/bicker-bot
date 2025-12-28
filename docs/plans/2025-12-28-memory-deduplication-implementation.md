# Memory Deduplication Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement two-threshold memory deduplication to prevent duplicate/near-duplicate memories from accumulating.

**Architecture:** Pre-store similarity check with LLM-assisted merging for ambiguous cases. Batch cleanup mode for existing duplicates. Thresholds: >=0.95 auto-replace, 0.90-0.95 LLM merge, <0.90 add as new.

**Tech Stack:** ChromaDB, Gemini Flash (for merges), Pydantic config, pytest

---

### Task 1: Add Dedup Config Options

**Files:**
- Modify: `src/bicker_bot/config.py:31-37`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_memory_config_dedup_defaults():
    """Test dedup config has sensible defaults."""
    from bicker_bot.config import MemoryConfig

    config = MemoryConfig()
    assert config.dedup_enabled is True
    assert config.dedup_upper_threshold == 0.95
    assert config.dedup_lower_threshold == 0.90


def test_memory_config_dedup_validation():
    """Test dedup thresholds are validated."""
    from bicker_bot.config import MemoryConfig
    import pytest

    # Valid config
    config = MemoryConfig(dedup_upper_threshold=0.95, dedup_lower_threshold=0.90)
    assert config.dedup_upper_threshold == 0.95

    # Invalid: threshold > 1.0
    with pytest.raises(ValueError):
        MemoryConfig(dedup_upper_threshold=1.5)

    # Invalid: threshold < 0.0
    with pytest.raises(ValueError):
        MemoryConfig(dedup_lower_threshold=-0.1)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_memory_config_dedup_defaults -v`
Expected: FAIL with "AttributeError: dedup_enabled"

**Step 3: Write minimal implementation**

In `src/bicker_bot/config.py`, update `MemoryConfig`:

```python
class MemoryConfig(BaseModel):
    """Memory/RAG configuration."""

    chroma_path: Path = Path("./data/chroma")
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    high_intensity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7

    # Deduplication settings
    dedup_enabled: bool = True
    dedup_upper_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.95
    dedup_lower_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.90
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py::test_memory_config_dedup_defaults tests/test_config.py::test_memory_config_dedup_validation -v`
Expected: PASS

**Step 5: Commit**

```bash
jj commit -m "feat(config): add memory deduplication config options"
```

---

### Task 2: Add MemoryStore.find_similar()

**Files:**
- Modify: `src/bicker_bot/memory/store.py`
- Test: `tests/test_memory.py`

**Step 1: Write the failing test**

Add to `tests/test_memory.py` in `TestMemoryStore` class:

```python
def test_find_similar_returns_match(self, memory_store: MemoryStore):
    """Test finding similar memories above threshold."""
    memory_store.add(Memory(content="Alice likes cats"))
    memory_store.add(Memory(content="Bob likes dogs"))

    # Search for something similar to cats
    result = memory_store.find_similar("Alice really loves cats", threshold=0.5)

    assert result is not None
    assert "cats" in result.memory.content.lower()


def test_find_similar_returns_none_below_threshold(self, memory_store: MemoryStore):
    """Test that find_similar returns None when no match above threshold."""
    memory_store.add(Memory(content="Alice likes cats"))

    # Very high threshold should return None
    result = memory_store.find_similar("Something completely different", threshold=0.99)

    assert result is None


def test_find_similar_empty_store(self, memory_store: MemoryStore):
    """Test find_similar on empty store returns None."""
    result = memory_store.find_similar("Any query", threshold=0.5)
    assert result is None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_memory.py::TestMemoryStore::test_find_similar_returns_match -v`
Expected: FAIL with "AttributeError: 'MemoryStore' object has no attribute 'find_similar'"

**Step 3: Write minimal implementation**

Add to `src/bicker_bot/memory/store.py` in the `MemoryStore` class:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_memory.py::TestMemoryStore::test_find_similar_returns_match tests/test_memory.py::TestMemoryStore::test_find_similar_returns_none_below_threshold tests/test_memory.py::TestMemoryStore::test_find_similar_empty_store -v`
Expected: PASS

**Step 5: Commit**

```bash
jj commit -m "feat(memory): add MemoryStore.find_similar() for dedup lookup"
```

---

### Task 3: Add MemoryStore.update()

**Files:**
- Modify: `src/bicker_bot/memory/store.py`
- Test: `tests/test_memory.py`

**Step 1: Write the failing test**

Add to `tests/test_memory.py` in `TestMemoryStore` class:

```python
def test_update_memory_content(self, memory_store: MemoryStore):
    """Test updating a memory's content."""
    memory = Memory(content="Original content", user="alice", intensity=0.8)
    memory_store.add(memory)

    memory_store.update(memory.id, new_content="Updated content")

    # Verify the update
    results = memory_store.search("Updated content")
    assert len(results) > 0
    assert "Updated" in results[0].memory.content


def test_update_preserves_metadata(self, memory_store: MemoryStore):
    """Test that update preserves user and intensity."""
    memory = Memory(content="Original", user="bob", intensity=0.9, memory_type=MemoryType.OPINION)
    memory_store.add(memory)

    memory_store.update(memory.id, new_content="New content")

    # Get the memory back
    memories = memory_store.get_user_memories("bob")
    assert len(memories) == 1
    assert memories[0].user == "bob"
    assert memories[0].intensity == 0.9
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_memory.py::TestMemoryStore::test_update_memory_content -v`
Expected: FAIL with "AttributeError: 'MemoryStore' object has no attribute 'update'"

**Step 3: Write minimal implementation**

Add to `src/bicker_bot/memory/store.py` in the `MemoryStore` class:

```python
def update(self, memory_id: str, new_content: str) -> bool:
    """Update a memory's content while preserving metadata.

    Args:
        memory_id: ID of memory to update
        new_content: New content text

    Returns:
        True if update succeeded, False if memory not found
    """
    # Get existing memory to preserve metadata
    try:
        existing = self._collection.get(
            ids=[memory_id],
            include=["metadatas"],
        )
    except Exception as e:
        logger.error(f"Failed to get memory {memory_id} for update: {e}")
        return False

    if not existing["ids"]:
        return False

    metadata = existing["metadatas"][0] if existing["metadatas"] else {}

    # Update the document
    self._collection.update(
        ids=[memory_id],
        documents=[new_content],
        metadatas=[metadata],
    )

    logger.info(f"MEMORY_UPDATE: id={memory_id[:8]}... content updated")
    return True
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_memory.py::TestMemoryStore::test_update_memory_content tests/test_memory.py::TestMemoryStore::test_update_preserves_metadata -v`
Expected: PASS

**Step 5: Commit**

```bash
jj commit -m "feat(memory): add MemoryStore.update() for in-place content updates"
```

---

### Task 4: Create MemoryDeduplicator Class

**Files:**
- Create: `src/bicker_bot/memory/deduplicator.py`
- Modify: `src/bicker_bot/memory/__init__.py`
- Test: `tests/test_deduplication.py`

**Step 1: Write the failing test**

Create `tests/test_deduplication.py`:

```python
"""Tests for memory deduplication."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bicker_bot.config import MemoryConfig
from bicker_bot.memory.store import Memory, MemoryStore, MemoryType


class MockEmbeddingFunction:
    """Mock embedding function for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @staticmethod
    def name() -> str:
        return "mock"

    def __call__(self, input: list[str]) -> list[list[float]]:
        result = []
        for text in input:
            h = hash(text)
            embedding = [(h >> i) % 100 / 100.0 for i in range(self._dimension)]
            result.append(embedding)
        return result

    def embed_query(self, query: str) -> list[float]:
        return self([query])[0]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self(documents)


class TestMemoryDeduplicator:
    """Tests for MemoryDeduplicator."""

    @pytest.fixture
    def memory_store(self, tmp_path: Path) -> MemoryStore:
        """Create a memory store with mock embeddings."""
        config = MemoryConfig(
            chroma_path=tmp_path / "chroma",
            embedding_model="test-model",
        )

        with patch(
            "bicker_bot.memory.store.LocalEmbeddingFunction",
            return_value=MockEmbeddingFunction(),
        ):
            store = MemoryStore(config)
            yield store

    @pytest.fixture
    def deduplicator(self, memory_store: MemoryStore):
        """Create a deduplicator with mocked LLM."""
        from bicker_bot.memory.deduplicator import MemoryDeduplicator

        dedup = MemoryDeduplicator(
            store=memory_store,
            api_key="test-key",
            upper_threshold=0.95,
            lower_threshold=0.90,
        )
        return dedup

    def test_deduplicator_creation(self, deduplicator):
        """Test deduplicator can be instantiated."""
        from bicker_bot.memory.deduplicator import MemoryDeduplicator

        assert isinstance(deduplicator, MemoryDeduplicator)

    @pytest.mark.asyncio
    async def test_check_and_merge_new_memory(self, deduplicator, memory_store):
        """Test that new memories pass through unchanged."""
        memory = Memory(content="Brand new unique content")

        result = await deduplicator.check_and_merge(memory)

        assert result.content == memory.content
        assert result.id == memory.id

    @pytest.mark.asyncio
    async def test_check_and_merge_auto_replace(self, deduplicator, memory_store):
        """Test auto-replace when similarity >= upper threshold."""
        # Add existing memory
        old_memory = Memory(content="Alice likes cats")
        memory_store.add(old_memory)

        # Mock find_similar to return high similarity
        with patch.object(memory_store, 'find_similar') as mock_find:
            from bicker_bot.memory.store import SearchResult
            mock_find.return_value = SearchResult(memory=old_memory, distance=0.02)  # 0.98 similarity

            new_memory = Memory(content="Alice likes cats a lot")
            result = await deduplicator.check_and_merge(new_memory)

            # Should return new memory (old one should be deleted)
            assert result.content == new_memory.content

    @pytest.mark.asyncio
    async def test_check_and_merge_llm_merge(self, deduplicator, memory_store):
        """Test LLM merge when similarity in gray zone."""
        old_memory = Memory(content="Alice has a cat")
        memory_store.add(old_memory)

        # Mock find_similar to return gray zone similarity
        with patch.object(memory_store, 'find_similar') as mock_find:
            from bicker_bot.memory.store import SearchResult
            mock_find.return_value = SearchResult(memory=old_memory, distance=0.08)  # 0.92 similarity

            # Mock the LLM merge call
            with patch.object(deduplicator, '_merge_with_llm', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = "Alice has a tabby cat named Whiskers"

                new_memory = Memory(content="Alice's cat is named Whiskers")
                result = await deduplicator.check_and_merge(new_memory)

                assert result.content == "Alice has a tabby cat named Whiskers"
                mock_llm.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deduplication.py::TestMemoryDeduplicator::test_deduplicator_creation -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'bicker_bot.memory.deduplicator'"

**Step 3: Write minimal implementation**

Create `src/bicker_bot/memory/deduplicator.py`:

```python
"""Memory deduplication using similarity thresholds and LLM merging."""

import logging
from dataclasses import dataclass

from google import genai
from google.genai import types

from bicker_bot.memory.store import Memory, MemoryStore

logger = logging.getLogger(__name__)


MERGE_PROMPT = """You are merging two similar memories about the same topic.
Synthesize them into a single, comprehensive memory.

Rules:
- Combine all distinct information from both
- If there's conflicting information, prefer the NEWER memory
- Keep the result concise but complete (1-3 sentences)
- Preserve the user attribution if present

OLDER memory (recorded {timestamp_old}):
{content_old}

NEWER memory (recorded {timestamp_new}):
{content_new}

Return only the merged memory text, nothing else."""


@dataclass
class DeduplicationReport:
    """Report from batch deduplication."""

    total_checked: int = 0
    auto_replaced: int = 0
    llm_merged: int = 0
    kept_new: int = 0
    errors: int = 0


class MemoryDeduplicator:
    """Handles memory deduplication at write-time and batch cleanup."""

    def __init__(
        self,
        store: MemoryStore,
        api_key: str,
        model: str = "gemini-3-flash-preview",
        upper_threshold: float = 0.95,
        lower_threshold: float = 0.90,
    ):
        """Initialize the deduplicator.

        Args:
            store: Memory store to deduplicate
            api_key: Google AI API key for LLM merging
            model: Model to use for merging
            upper_threshold: Above this, auto-replace (newer wins)
            lower_threshold: Above this, LLM merges. Below, add as new.
        """
        self._store = store
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._upper_threshold = upper_threshold
        self._lower_threshold = lower_threshold

    async def check_and_merge(self, new_memory: Memory) -> Memory:
        """Check for similar existing memory and merge if needed.

        Args:
            new_memory: New memory to potentially deduplicate

        Returns:
            The memory to store (original, merged, or replacement)
        """
        # Find most similar existing memory
        similar = self._store.find_similar(
            new_memory.to_chroma_document(),
            threshold=self._lower_threshold,
        )

        if similar is None:
            # No similar memory found, add as new
            logger.debug(f"DEDUP_CHECK: No similar memory found, adding as new")
            return new_memory

        similarity = similar.similarity

        if similarity >= self._upper_threshold:
            # Very similar - auto-replace with newer
            logger.info(
                f"DEDUP_REPLACE: sim={similarity:.3f} >= {self._upper_threshold}, "
                f"replacing '{similar.memory.content[:50]}...' with newer"
            )
            self._store.delete(similar.memory.id)
            return new_memory

        # Gray zone - LLM merge
        logger.info(
            f"DEDUP_MERGE: sim={similarity:.3f} in [{self._lower_threshold}, {self._upper_threshold}), "
            f"invoking LLM merge"
        )

        try:
            merged_content = await self._merge_with_llm(similar.memory, new_memory)
            self._store.delete(similar.memory.id)

            # Create merged memory with newer metadata but max intensity
            merged_memory = Memory(
                content=merged_content,
                user=new_memory.user or similar.memory.user,
                memory_type=new_memory.memory_type,
                intensity=max(new_memory.intensity, similar.memory.intensity),
                timestamp=new_memory.timestamp,
            )

            logger.info(f"DEDUP_MERGED: '{merged_content[:60]}...'")
            return merged_memory

        except Exception as e:
            logger.warning(f"DEDUP_FAIL: LLM merge failed ({e}), falling back to replace")
            self._store.delete(similar.memory.id)
            return new_memory

    async def _merge_with_llm(self, old_memory: Memory, new_memory: Memory) -> str:
        """Use LLM to merge two similar memories.

        Args:
            old_memory: Existing memory
            new_memory: New memory to merge in

        Returns:
            Merged content string
        """
        prompt = MERGE_PROMPT.format(
            timestamp_old=old_memory.timestamp.isoformat(),
            content_old=old_memory.content,
            timestamp_new=new_memory.timestamp.isoformat(),
            content_new=new_memory.content,
        )

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=500,
            ),
        )

        return response.text.strip()

    async def deduplicate_all(self, dry_run: bool = False) -> DeduplicationReport:
        """Run batch deduplication on all existing memories.

        Args:
            dry_run: If True, report what would happen without making changes

        Returns:
            Report of deduplication actions
        """
        import numpy as np

        report = DeduplicationReport()

        # Get all memories with embeddings
        count = self._store._collection.count()
        if count == 0:
            return report

        results = self._store._collection.get(
            include=["documents", "metadatas", "embeddings"],
            limit=count,
        )

        ids = results["ids"]
        documents = results["documents"]
        metadatas = results["metadatas"]
        embeddings = np.array(results["embeddings"])

        report.total_checked = len(ids)

        # Build clusters using union-find
        parent = list(range(len(ids)))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Find all pairs above lower threshold
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                # Cosine similarity
                sim = float(np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                ))
                if sim >= self._lower_threshold:
                    union(i, j)

        # Group by cluster
        clusters: dict[int, list[int]] = {}
        for i in range(len(ids)):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)

        # Process clusters with more than one member
        for root, members in clusters.items():
            if len(members) == 1:
                continue

            # Sort by timestamp (oldest first)
            def get_timestamp(idx: int) -> str:
                return metadatas[idx].get("timestamp", "")

            members.sort(key=get_timestamp)

            # Get max similarity in cluster
            max_sim = 0.0
            for i, idx_i in enumerate(members):
                for idx_j in members[i + 1:]:
                    sim = float(np.dot(embeddings[idx_i], embeddings[idx_j]) / (
                        np.linalg.norm(embeddings[idx_i]) * np.linalg.norm(embeddings[idx_j])
                    ))
                    max_sim = max(max_sim, sim)

            newest_idx = members[-1]
            to_delete = members[:-1]

            if max_sim >= self._upper_threshold:
                # Auto-replace: keep newest
                if not dry_run:
                    for idx in to_delete:
                        self._store.delete(ids[idx])
                report.auto_replaced += len(to_delete)
                logger.info(
                    f"BATCH_DEDUP: Cluster of {len(members)} memories, "
                    f"max_sim={max_sim:.3f} >= {self._upper_threshold}, "
                    f"keeping newest, deleting {len(to_delete)}"
                )
            else:
                # LLM merge all in cluster
                try:
                    # Build memories from cluster
                    cluster_memories = []
                    for idx in members:
                        from datetime import datetime
                        from bicker_bot.memory.store import MemoryType

                        cluster_memories.append(Memory(
                            id=ids[idx],
                            content=documents[idx],
                            user=metadatas[idx].get("user") or None,
                            memory_type=MemoryType(metadatas[idx].get("type", "fact")),
                            intensity=float(metadatas[idx].get("intensity", 0.5)),
                            timestamp=datetime.fromisoformat(
                                metadatas[idx].get("timestamp", datetime.now().isoformat())
                            ),
                        ))

                    if not dry_run:
                        merged_content = await self._merge_cluster(cluster_memories)

                        # Delete all old memories
                        for idx in members:
                            self._store.delete(ids[idx])

                        # Add merged memory
                        merged = Memory(
                            content=merged_content,
                            user=cluster_memories[-1].user,
                            memory_type=cluster_memories[-1].memory_type,
                            intensity=max(m.intensity for m in cluster_memories),
                            timestamp=cluster_memories[-1].timestamp,
                        )
                        self._store.add(merged)

                    report.llm_merged += len(members)
                    logger.info(
                        f"BATCH_DEDUP: Cluster of {len(members)} memories merged via LLM"
                    )

                except Exception as e:
                    logger.error(f"BATCH_DEDUP: Merge failed for cluster: {e}")
                    report.errors += 1

        return report

    async def _merge_cluster(self, memories: list[Memory]) -> str:
        """Merge a cluster of similar memories using LLM.

        Args:
            memories: List of memories to merge (sorted oldest to newest)

        Returns:
            Merged content string
        """
        memories_text = "\n\n".join(
            f"Memory {i+1} (recorded {m.timestamp.isoformat()}):\n{m.content}"
            for i, m in enumerate(memories)
        )

        prompt = f"""You are merging {len(memories)} similar memories about the same topic.
Synthesize them into a single, comprehensive memory.

Rules:
- Combine all distinct information from all memories
- If there's conflicting information, prefer the NEWER memories
- Keep the result concise but complete (1-3 sentences)
- Preserve user attributions if present

{memories_text}

Return only the merged memory text, nothing else."""

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=500,
            ),
        )

        return response.text.strip()
```

**Step 4: Update exports**

In `src/bicker_bot/memory/__init__.py`, add:

```python
from .deduplicator import DeduplicationReport, MemoryDeduplicator
```

And update `__all__`:

```python
__all__ = [
    "BotIdentity",
    "BotSelector",
    "DeduplicationReport",
    "ExtractionResult",
    "LocalEmbeddingFunction",
    "Memory",
    "MemoryDeduplicator",
    "MemoryExtractor",
    "MemoryStore",
    "MemoryType",
    "SearchResult",
    "SelectionResult",
]
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_deduplication.py -v`
Expected: PASS

**Step 6: Commit**

```bash
jj commit -m "feat(memory): add MemoryDeduplicator for write-time and batch dedup"
```

---

### Task 5: Integrate Deduplicator with Extractor

**Files:**
- Modify: `src/bicker_bot/memory/extractor.py`
- Test: `tests/test_deduplication.py`

**Step 1: Write the failing test**

Add to `tests/test_deduplication.py`:

```python
class TestExtractorIntegration:
    """Tests for extractor + deduplicator integration."""

    @pytest.mark.asyncio
    async def test_extractor_uses_deduplicator(self, tmp_path: Path):
        """Test that extractor deduplicates before storing."""
        from unittest.mock import AsyncMock, patch, MagicMock

        config = MemoryConfig(
            chroma_path=tmp_path / "chroma",
            embedding_model="test-model",
            dedup_enabled=True,
        )

        with patch(
            "bicker_bot.memory.store.LocalEmbeddingFunction",
            return_value=MockEmbeddingFunction(),
        ):
            store = MemoryStore(config)

        with patch("bicker_bot.memory.extractor.MemoryDeduplicator") as MockDedup:
            mock_dedup_instance = MagicMock()
            mock_dedup_instance.check_and_merge = AsyncMock(side_effect=lambda m: m)
            MockDedup.return_value = mock_dedup_instance

            from bicker_bot.memory.extractor import MemoryExtractor

            extractor = MemoryExtractor(
                api_key="test-key",
                memory_store=store,
                dedup_config=config,
            )

            assert extractor._deduplicator is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deduplication.py::TestExtractorIntegration::test_extractor_uses_deduplicator -v`
Expected: FAIL with "TypeError: MemoryExtractor.__init__() got an unexpected keyword argument 'dedup_config'"

**Step 3: Write minimal implementation**

Update `src/bicker_bot/memory/extractor.py`:

Add import at top:
```python
from bicker_bot.config import MemoryConfig
from bicker_bot.memory.deduplicator import MemoryDeduplicator
```

Update `__init__`:
```python
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
```

Update `extract_and_store` method - replace the memory storage section:

```python
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
    # ... rest of logging
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_deduplication.py::TestExtractorIntegration -v`
Expected: PASS

**Step 5: Commit**

```bash
jj commit -m "feat(memory): integrate deduplicator with memory extractor"
```

---

### Task 6: Update Extraction Prompt for Longer Memories

**Files:**
- Modify: `src/bicker_bot/memory/extractor.py`

**Step 1: Update the prompt**

In `src/bicker_bot/memory/extractor.py`, update `EXTRACTION_SYSTEM_PROMPT`:

```python
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
```

**Step 2: Run existing tests to ensure no regression**

Run: `uv run pytest tests/test_memory.py -v`
Expected: PASS

**Step 3: Commit**

```bash
jj commit -m "feat(memory): update extraction prompt for longer, more contextual memories"
```

---

### Task 7: Add CLI Flags for Batch Deduplication

**Files:**
- Modify: `src/bicker_bot/main.py`

**Step 1: Add CLI arguments**

In `src/bicker_bot/main.py`, update the argument parser in `main()`:

```python
parser.add_argument(
    "--deduplicate-memories",
    action="store_true",
    help="Run batch deduplication on startup before connecting to IRC",
)
parser.add_argument(
    "--dedup-dry-run",
    action="store_true",
    help="Show what batch deduplication would do without making changes",
)
```

**Step 2: Update async_main signature and logic**

```python
async def async_main(
    config_path: str | None = None,
    debug: bool = False,
    debug_ai: bool = False,
    deduplicate_memories: bool = False,
    dedup_dry_run: bool = False,
) -> None:
```

Add after config loading, before "Starting Bicker-Bot":

```python
# Run batch deduplication if requested
if deduplicate_memories:
    from bicker_bot.memory import MemoryDeduplicator, MemoryStore

    logger.info("Running batch memory deduplication...")

    memory_store = MemoryStore(config.memory)
    google_key = config.llm.google_api_key
    if not google_key:
        logger.error("Google API key required for deduplication")
        sys.exit(1)

    deduplicator = MemoryDeduplicator(
        store=memory_store,
        api_key=google_key.get_secret_value(),
        upper_threshold=config.memory.dedup_upper_threshold,
        lower_threshold=config.memory.dedup_lower_threshold,
    )

    report = await deduplicator.deduplicate_all(dry_run=dedup_dry_run)

    mode = "DRY RUN" if dedup_dry_run else "COMPLETED"
    logger.info(
        f"Deduplication {mode}: "
        f"checked={report.total_checked}, "
        f"auto_replaced={report.auto_replaced}, "
        f"llm_merged={report.llm_merged}, "
        f"errors={report.errors}"
    )

    if dedup_dry_run:
        logger.info("Dry run complete, exiting without starting bot")
        return
```

Update the call in `main()`:

```python
asyncio.run(async_main(
    config_path=args.config,
    debug=args.debug,
    debug_ai=args.debug_ai,
    deduplicate_memories=args.deduplicate_memories,
    dedup_dry_run=args.dedup_dry_run,
))
```

**Step 3: Test manually**

Run: `uv run python -m bicker_bot.main --help`
Expected: Should show `--deduplicate-memories` and `--dedup-dry-run` options

**Step 4: Commit**

```bash
jj commit -m "feat(cli): add --deduplicate-memories and --dedup-dry-run flags"
```

---

### Task 8: Update Orchestrator to Pass Dedup Config

**Files:**
- Modify: `src/bicker_bot/orchestrator.py`

**Step 1: Find where MemoryExtractor is instantiated**

In `orchestrator.py`, update the `MemoryExtractor` instantiation to pass the config:

```python
self._memory_extractor = MemoryExtractor(
    api_key=google_key.get_secret_value(),
    memory_store=self._memory_store,
    dedup_config=config.memory,
)
```

**Step 2: Run full test suite**

Run: `uv run pytest tests/ -v --ignore=tests/test_integration.py`
Expected: PASS

**Step 3: Commit**

```bash
jj commit -m "feat(orchestrator): enable memory deduplication in extractor"
```

---

### Task 9: Add Integration Tests

**Files:**
- Add to: `tests/test_deduplication.py`

**Step 1: Add integration test for write-time dedup**

```python
class TestDeduplicationIntegration:
    """Integration tests for deduplication (requires mocked LLM)."""

    @pytest.fixture
    def full_setup(self, tmp_path: Path):
        """Set up store and deduplicator with controllable embeddings."""
        config = MemoryConfig(
            chroma_path=tmp_path / "chroma",
            embedding_model="test-model",
            dedup_upper_threshold=0.95,
            dedup_lower_threshold=0.90,
        )

        # Use embeddings that give predictable similarity
        class ControlledEmbeddings:
            def __init__(self):
                self._embeddings = {}

            @staticmethod
            def name() -> str:
                return "controlled"

            def set_embedding(self, text: str, embedding: list[float]):
                self._embeddings[text] = embedding

            def __call__(self, input: list[str]) -> list[list[float]]:
                return [self._embeddings.get(t, [0.0] * 384) for t in input]

            def embed_query(self, query: str) -> list[float]:
                return self([query])[0]

            def embed_documents(self, documents: list[str]) -> list[list[float]]:
                return self(documents)

        embedding_fn = ControlledEmbeddings()

        with patch(
            "bicker_bot.memory.store.LocalEmbeddingFunction",
            return_value=embedding_fn,
        ):
            store = MemoryStore(config)

        return store, embedding_fn, config

    @pytest.mark.asyncio
    async def test_auto_replace_very_similar(self, full_setup):
        """Test that very similar memories trigger auto-replace."""
        store, embedding_fn, config = full_setup

        from bicker_bot.memory.deduplicator import MemoryDeduplicator

        # Set up embeddings so they're very similar (cosine sim ~0.99)
        base = [1.0] + [0.0] * 383
        similar = [0.995] + [0.1] + [0.0] * 382

        embedding_fn.set_embedding("About alice: Alice likes cats", base)
        embedding_fn.set_embedding("About alice: Alice really likes cats", similar)

        # Add first memory
        old = Memory(content="Alice likes cats", user="alice")
        store.add(old)

        dedup = MemoryDeduplicator(
            store=store,
            api_key="test",
            upper_threshold=0.95,
            lower_threshold=0.90,
        )

        # Check and merge new memory
        new = Memory(content="Alice really likes cats", user="alice")

        with patch.object(store, 'find_similar') as mock_find:
            from bicker_bot.memory.store import SearchResult
            # Return high similarity match
            mock_find.return_value = SearchResult(memory=old, distance=0.01)

            result = await dedup.check_and_merge(new)

        # Should have replaced old with new
        assert result.content == new.content
        assert store.count() == 0  # Old was deleted, new not yet added
```

**Step 2: Run integration tests**

Run: `uv run pytest tests/test_deduplication.py -v`
Expected: PASS

**Step 3: Commit**

```bash
jj commit -m "test(memory): add deduplication integration tests"
```

---

### Task 10: Final Verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All PASS

**Step 2: Run linter**

Run: `uv run ruff check src/`
Expected: No errors (or fix any that appear)

**Step 3: Test batch dedup dry-run**

Run: `uv run python -m bicker_bot.main --deduplicate-memories --dedup-dry-run`
Expected: Shows deduplication report without making changes

**Step 4: Final commit**

```bash
jj commit -m "feat(memory): complete memory deduplication implementation"
```
