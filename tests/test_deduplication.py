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

    @pytest.mark.asyncio
    async def test_check_and_merge_below_threshold(self, deduplicator, memory_store):
        """Test that memories below lower threshold pass through unchanged."""
        old_memory = Memory(content="Alice likes cats")
        memory_store.add(old_memory)

        # Mock find_similar to return low similarity (None means below threshold)
        with patch.object(memory_store, 'find_similar') as mock_find:
            mock_find.return_value = None  # Below threshold

            new_memory = Memory(content="Bob likes dogs")
            result = await deduplicator.check_and_merge(new_memory)

            # Should return unchanged
            assert result.content == new_memory.content
            assert result.id == new_memory.id

    @pytest.mark.asyncio
    async def test_auto_replace_deletes_old_memory(self, deduplicator, memory_store):
        """Test that auto-replace actually deletes the old memory."""
        old_memory = Memory(content="Alice likes cats")
        memory_store.add(old_memory)

        with patch.object(memory_store, 'find_similar') as mock_find:
            from bicker_bot.memory.store import SearchResult
            mock_find.return_value = SearchResult(memory=old_memory, distance=0.02)  # 0.98 similarity

            with patch.object(memory_store, 'delete') as mock_delete:
                new_memory = Memory(content="Alice likes cats a lot")
                await deduplicator.check_and_merge(new_memory)

                mock_delete.assert_called_once_with(old_memory.id)

    @pytest.mark.asyncio
    async def test_llm_merge_preserves_metadata(self, deduplicator, memory_store):
        """Test that merged memory preserves important metadata."""
        from datetime import datetime

        old_memory = Memory(
            content="Alice has a cat",
            user="alice",
            intensity=0.6,
            memory_type=MemoryType.FACT,
        )
        memory_store.add(old_memory)

        with patch.object(memory_store, 'find_similar') as mock_find:
            from bicker_bot.memory.store import SearchResult
            mock_find.return_value = SearchResult(memory=old_memory, distance=0.08)  # 0.92 similarity

            with patch.object(deduplicator, '_merge_with_llm', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = "Alice has a tabby cat named Whiskers"

                new_memory = Memory(
                    content="Alice's cat is named Whiskers",
                    user="alice",
                    intensity=0.8,
                    memory_type=MemoryType.FACT,
                )
                result = await deduplicator.check_and_merge(new_memory)

                # Should preserve higher intensity
                assert result.intensity == 0.8
                # Should preserve user
                assert result.user == "alice"
                # Should use merged content
                assert result.content == "Alice has a tabby cat named Whiskers"


class TestDeduplicationReport:
    """Tests for DeduplicationReport."""

    def test_report_creation(self):
        """Test report can be created."""
        from bicker_bot.memory.deduplicator import DeduplicationReport

        report = DeduplicationReport(
            memories_scanned=100,
            clusters_found=5,
            memories_merged=10,
            memories_deleted=8,
        )

        assert report.memories_scanned == 100
        assert report.clusters_found == 5
        assert report.memories_merged == 10
        assert report.memories_deleted == 8


class TestBatchDeduplication:
    """Tests for batch deduplication mode."""

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

    @pytest.mark.asyncio
    async def test_deduplicate_all_empty_store(self, deduplicator, memory_store):
        """Test batch dedup on empty store."""
        from bicker_bot.memory.deduplicator import DeduplicationReport

        report = await deduplicator.deduplicate_all()

        assert isinstance(report, DeduplicationReport)
        assert report.memories_scanned == 0
        assert report.clusters_found == 0

    @pytest.mark.asyncio
    async def test_deduplicate_all_dry_run(self, deduplicator, memory_store):
        """Test dry run mode doesn't modify data."""
        # Add some memories
        memory_store.add(Memory(content="Alice likes cats"))
        memory_store.add(Memory(content="Bob likes dogs"))

        initial_count = memory_store.count()

        report = await deduplicator.deduplicate_all(dry_run=True)

        # Count should not change
        assert memory_store.count() == initial_count
