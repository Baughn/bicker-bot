# Memory Deduplication Design

**Date:** 2025-12-28
**Status:** Approved

## Problem

Duplicate and near-duplicate memories accumulate in ChromaDB, crowding out diverse information. Analysis of 1873 memories found:
- 254 pairs with similarity >= 0.98 (near-identical)
- 825 pairs with similarity >= 0.95
- 2646 pairs with similarity >= 0.90

## Solution

Two-threshold deduplication at write time, with LLM-assisted merging for ambiguous cases. Batch cleanup mode for existing duplicates.

```
New memory extracted
        |
        v
Query ChromaDB for similar (top 1)
        |
        +-- similarity >= 0.95 --> Delete old, insert new (auto-replace)
        |
        +-- similarity 0.90-0.95 --> LLM synthesizes merged memory --> Delete old, insert merged
        |
        +-- similarity < 0.90 --> Insert as new
```

## Thresholds

| Similarity | Action | Rationale |
|------------|--------|-----------|
| >= 0.95 | Auto-replace with newer | Minor rewording, newer is likely more detailed |
| 0.90-0.95 | LLM merge | Same fact expressed differently, worth combining |
| < 0.90 | Add as new | Sufficiently distinct topics |

Thresholds are configurable via `config.yaml`.

## Components

### MemoryDeduplicator (`memory/deduplicator.py`)

New class that handles both write-time and batch deduplication:

```python
class MemoryDeduplicator:
    def __init__(
        self,
        store: MemoryStore,
        api_key: str,
        model: str = "gemini-3-flash-preview",
        upper_threshold: float = 0.95,
        lower_threshold: float = 0.90,
    ):
        ...

    async def check_and_merge(self, new_memory: Memory) -> Memory:
        """Write-time dedup: check one new memory against existing."""

    async def deduplicate_all(self, dry_run: bool = False) -> DeduplicationReport:
        """Batch mode: find and merge all duplicate pairs."""
```

### LLM Merge Prompt

```
You are merging two similar memories about the same topic.
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

Return only the merged memory text, nothing else.
```

For batch mode with 3+ similar memories, all memories in the cluster are provided sorted oldest to newest. Merged memory inherits metadata from newest, with intensity set to max of all.

### MemoryStore Additions

```python
def find_similar(self, content: str, threshold: float = 0.90) -> SearchResult | None:
    """Find the most similar existing memory above threshold."""

def update(self, memory_id: str, new_content: str) -> None:
    """Update existing memory content in place."""
```

### Configuration

Add to `MemoryConfig` in `config.py`:

```python
dedup_enabled: bool = True
dedup_upper_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.95
dedup_lower_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.90
```

Example `config.yaml`:

```yaml
memory:
  dedup_enabled: true
  dedup_upper_threshold: 0.95
  dedup_lower_threshold: 0.90
```

### CLI Flags

| Flag | Purpose |
|------|---------|
| `--deduplicate-memories` | Run batch dedup before starting |
| `--dedup-dry-run` | Show what would be merged/deleted without acting |

### Extractor Integration

Modified flow in `extract_and_store()`:

```python
memories = [parse extracted memories from LLM]
deduplicated = []
for memory in memories:
    result = await self._deduplicator.check_and_merge(memory)
    deduplicated.append(result)
self._memory_store.add_batch(deduplicated)
```

## Longer Memories

Update extraction prompt to encourage richer context:

```
For each memory, provide:
- content: The fact to remember. Include relevant context such as:
  - When it was mentioned (if notable)
  - Why it matters or how it came up
  - Related details that distinguish this from similar facts
  Aim for 1-3 sentences rather than fragments.
```

Example improvement:

| Before | After |
|--------|-------|
| "Alice likes cats" | "Alice mentioned she adopted a rescue cat in 2023 after her previous cat passed away" |

## Error Handling

| Scenario | Behavior |
|----------|----------|
| ChromaDB query fails during dedup check | Log warning, store memory anyway (fail open) |
| LLM merge call fails | Keep newer memory, delete older (fall back to auto-replace) |
| Similarity exactly at threshold | Treat as "above" (inclusive lower bound) |
| New memory similar to multiple existing | Write-time: single match only. Batch: cluster all. |
| Empty memory after merge | Discard, log warning |
| Batch finds transitive clusters (A~B, B~C, A~C) | Union-find to group, merge entire cluster |

## Logging

```
DEBUG: DEDUP_CHECK: "Alice has a cat" closest match sim=0.93 -> LLM merge
INFO:  DEDUP_MERGE: 2 memories -> "Alice has a tabby cat named Whiskers"
WARN:  DEDUP_FAIL: LLM merge failed, falling back to replace
```

## Files Changed

**New:**
- `memory/deduplicator.py` - MemoryDeduplicator class

**Modified:**
- `config.py` - Add dedup config options
- `memory/store.py` - Add `find_similar()` and `update()` methods
- `memory/extractor.py` - Integrate deduplicator, update extraction prompt
- `memory/__init__.py` - Export MemoryDeduplicator
- `orchestrator.py` - Add CLI flags for batch mode

## Testing

| Test | Coverage |
|------|----------|
| Unit: threshold logic | Auto-replace vs LLM-merge vs add-new decisions |
| Unit: cluster building | Union-find groups transitive similarities |
| Integration: write-time dedup | Similar memory triggers merge |
| Integration: batch dedup | Cleans existing duplicates |
| Integration: dry-run | Reports without modifying |

## Implementation Order

1. Config changes
2. `MemoryStore.find_similar()` and `update()`
3. `MemoryDeduplicator` core logic
4. Extractor integration + prompt update
5. CLI flags for batch mode
6. Tests
