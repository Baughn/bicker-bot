#!/usr/bin/env python3
"""Analyze pairwise similarity of all memories to find dedup thresholds."""

from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

# Default config values
CHROMA_PATH = Path("./data/chroma")
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    chroma_path = CHROMA_PATH

    if not chroma_path.exists():
        print(f"ChromaDB path {chroma_path} does not exist")
        return

    # Connect to ChromaDB
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False),
    )

    # Get collection without embedding function - we just need stored embeddings
    collection = client.get_collection(name="memories")

    count = collection.count()
    print(f"Total memories: {count}")

    if count == 0:
        print("No memories to analyze")
        return

    # Get all memories with embeddings
    print("Fetching all memories with embeddings...")
    results = collection.get(
        include=["documents", "metadatas", "embeddings"],
        limit=count,
    )

    ids = results["ids"]
    documents = results["documents"]
    metadatas = results["metadatas"]
    embeddings = np.array(results["embeddings"])

    print(f"Loaded {len(ids)} memories with {embeddings.shape[1]}-dim embeddings")

    # Compute all pairwise similarities
    print("\nComputing pairwise similarities...")
    n = len(ids)
    similarities = []
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)
            pairs.append((i, j, sim))

    similarities = np.array(similarities)
    print(f"Computed {len(similarities)} pairwise similarities")

    # Statistics
    print("\n" + "=" * 60)
    print("SIMILARITY DISTRIBUTION")
    print("=" * 60)
    print(f"Min:    {similarities.min():.4f}")
    print(f"Max:    {similarities.max():.4f}")
    print(f"Mean:   {similarities.mean():.4f}")
    print(f"Median: {np.median(similarities):.4f}")
    print(f"Std:    {similarities.std():.4f}")

    # Percentiles
    print("\nPercentiles:")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
        val = np.percentile(similarities, p)
        print(f"  {p:5.1f}%: {val:.4f}")

    # Histogram buckets
    print("\nHistogram (similarity buckets):")
    buckets = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.85),
               (0.85, 0.9), (0.9, 0.95), (0.95, 0.98), (0.98, 1.0)]
    for low, high in buckets:
        count_in_bucket = np.sum((similarities >= low) & (similarities < high))
        pct = 100 * count_in_bucket / len(similarities)
        bar = "#" * int(pct / 2)
        print(f"  [{low:.2f}-{high:.2f}): {count_in_bucket:5d} ({pct:5.1f}%) {bar}")

    # Sort pairs by similarity (highest first)
    pairs.sort(key=lambda x: x[2], reverse=True)

    # Show examples at different similarity levels
    print("\n" + "=" * 60)
    print("EXAMPLE PAIRS AT DIFFERENT SIMILARITY LEVELS")
    print("=" * 60)

    def show_pair(i: int, j: int, sim: float):
        doc_i = documents[i]
        doc_j = documents[j]
        user_i = metadatas[i].get("user", "")
        user_j = metadatas[j].get("user", "")
        print(f"\nSimilarity: {sim:.4f}")
        print(f"  A [{user_i}]: {doc_i}")
        print(f"  B [{user_j}]: {doc_j}")

    # Find pairs at specific similarity levels
    thresholds = [0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70]

    for threshold in thresholds:
        print(f"\n--- Pairs with similarity >= {threshold} ---")
        shown = 0
        for i, j, sim in pairs:
            if sim >= threshold and (threshold == thresholds[0] or sim < thresholds[thresholds.index(threshold) - 1]):
                show_pair(i, j, sim)
                shown += 1
                if shown >= 3:
                    break
        if shown == 0:
            print("  (no pairs in this range)")

    # Count potential duplicates at various thresholds
    print("\n" + "=" * 60)
    print("POTENTIAL DUPLICATES AT VARIOUS THRESHOLDS")
    print("=" * 60)
    for threshold in [0.98, 0.95, 0.92, 0.90, 0.85, 0.80]:
        dup_count = np.sum(similarities >= threshold)
        print(f"  >= {threshold}: {dup_count} pairs")


if __name__ == "__main__":
    main()
