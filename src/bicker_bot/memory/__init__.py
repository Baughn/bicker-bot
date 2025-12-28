"""Memory and RAG system."""

from .embeddings import LocalEmbeddingFunction
from .extractor import ExtractionResult, MemoryExtractor
from .selector import BotIdentity, BotSelector, SelectionResult
from .store import Memory, MemoryStore, MemoryType, SearchResult

# Lazy imports to avoid circular dependency
# (deduplicator imports from store, which imports from core.logging,
# which triggers core/__init__ which imports context, which imports store)


def __getattr__(name: str):
    """Lazy import for MemoryDeduplicator and DeduplicationReport."""
    if name == "MemoryDeduplicator":
        from .deduplicator import MemoryDeduplicator

        return MemoryDeduplicator
    if name == "DeduplicationReport":
        from .deduplicator import DeduplicationReport

        return DeduplicationReport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
