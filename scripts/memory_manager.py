#!/usr/bin/env python3
"""CLI tool for browsing and managing ChromaDB memories."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chromadb
from chromadb.config import Settings
from tabulate import tabulate


def get_collection(chroma_path: str):
    """Get the memories collection."""
    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        return client.get_collection("memories")
    except ValueError:
        print(f"No 'memories' collection found in {chroma_path}")
        sys.exit(1)


def cmd_list(args):
    """List all memories with optional filters."""
    collection = get_collection(args.chroma_path)

    # Build where clause
    where = None
    where_clauses = []

    if args.user:
        where_clauses.append({"user": {"$eq": args.user}})
    if args.type:
        where_clauses.append({"type": {"$eq": args.type}})
    if args.min_intensity is not None:
        where_clauses.append({"intensity": {"$gte": args.min_intensity}})
    if args.max_intensity is not None:
        where_clauses.append({"intensity": {"$lte": args.max_intensity}})

    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}

    # Fetch all matching records
    results = collection.get(
        where=where,
        limit=args.limit,
        include=["documents", "metadatas"],
    )

    if not results["ids"]:
        print("No memories found matching criteria.")
        return

    # Format for display
    rows = []
    for i, mem_id in enumerate(results["ids"]):
        meta = results["metadatas"][i] if results["metadatas"] else {}
        doc = results["documents"][i] if results["documents"] else ""

        # Truncate content for display
        content = doc[:80] + "..." if len(doc) > 80 else doc

        rows.append([
            mem_id[:12] + "...",
            meta.get("user", ""),
            meta.get("type", ""),
            f"{meta.get('intensity', 0):.2f}",
            meta.get("timestamp", "")[:19],  # Trim to datetime
            content,
        ])

    print(f"\nFound {len(rows)} memories:\n")
    print(tabulate(
        rows,
        headers=["ID", "User", "Type", "Intensity", "Timestamp", "Content"],
        tablefmt="simple",
    ))
    print()


def cmd_search(args):
    """Search memories by text query."""
    collection = get_collection(args.chroma_path)

    # Import embedding function directly to avoid circular imports
    from sentence_transformers import SentenceTransformer

    model_name = "nomic-ai/nomic-embed-text-v1.5"
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # Nomic embed expects "search_query: " prefix for queries
    query_text = f"search_query: {args.query}"
    query_embedding = model.encode(query_text, convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=args.limit,
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"] or not results["ids"][0]:
        print("No results found.")
        return

    rows = []
    for i, mem_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        doc = results["documents"][0][i] if results["documents"] else ""
        dist = results["distances"][0][i] if results["distances"] else 0

        # Truncate content for display
        content = doc[:80] + "..." if len(doc) > 80 else doc
        similarity = max(0, 1 - dist / 2)

        rows.append([
            mem_id[:12] + "...",
            f"{similarity:.3f}",
            meta.get("user", ""),
            meta.get("type", ""),
            content,
        ])

    print(f"\nSearch results for '{args.query}':\n")
    print(tabulate(
        rows,
        headers=["ID", "Similarity", "User", "Type", "Content"],
        tablefmt="simple",
    ))
    print()


def find_by_partial_id(collection, partial_id: str) -> list[str]:
    """Find memory IDs that start with the given partial ID."""
    all_ids = collection.get()["ids"]
    return [mid for mid in all_ids if mid.startswith(partial_id)]


def cmd_show(args):
    """Show full details of a specific memory."""
    collection = get_collection(args.chroma_path)

    # Support partial ID matching
    matching_ids = find_by_partial_id(collection, args.id)

    if not matching_ids:
        print(f"No memory found matching '{args.id}'")
        return

    if len(matching_ids) > 1:
        print(f"Multiple matches for '{args.id}':")
        for mid in matching_ids[:10]:
            print(f"  {mid}")
        if len(matching_ids) > 10:
            print(f"  ... and {len(matching_ids) - 10} more")
        return

    results = collection.get(
        ids=matching_ids,
        include=["documents", "metadatas"],
    )

    if not results["ids"]:
        print(f"Memory {args.id} not found.")
        return

    meta = results["metadatas"][0] if results["metadatas"] else {}
    doc = results["documents"][0] if results["documents"] else ""

    print(f"\n{'='*60}")
    print(f"ID:        {results['ids'][0]}")
    print(f"User:      {meta.get('user', '(none)')}")
    print(f"Type:      {meta.get('type', 'unknown')}")
    print(f"Intensity: {meta.get('intensity', 0):.2f}")
    print(f"Timestamp: {meta.get('timestamp', 'unknown')}")
    print(f"{'='*60}")
    print(f"Content:\n{doc}")
    print(f"{'='*60}\n")


def cmd_delete(args):
    """Delete memories by ID or filter."""
    collection = get_collection(args.chroma_path)

    if args.id:
        # Delete by ID (supports partial matching)
        matching_ids = find_by_partial_id(collection, args.id)

        if not matching_ids:
            print(f"No memory found matching '{args.id}'")
            return

        if len(matching_ids) > 1:
            print(f"Multiple matches for '{args.id}':")
            for mid in matching_ids[:10]:
                result = collection.get(ids=[mid], include=["documents"])
                doc = result["documents"][0][:60] + "..." if result["documents"] else ""
                print(f"  {mid}: {doc}")
            if len(matching_ids) > 10:
                print(f"  ... and {len(matching_ids) - 10} more")
            print("Use a more specific ID prefix.")
            return

        mem_id = matching_ids[0]
        if not args.force:
            results = collection.get(ids=[mem_id], include=["documents"])
            if results["ids"]:
                print(f"Will delete: {results['documents'][0][:80]}...")
                confirm = input("Confirm delete? [y/N]: ")
                if confirm.lower() != "y":
                    print("Cancelled.")
                    return

        collection.delete(ids=[mem_id])
        print(f"Deleted memory {mem_id}")

    elif args.user:
        # Delete all memories for a user
        results = collection.get(
            where={"user": {"$eq": args.user}},
            include=["documents"],
        )

        if not results["ids"]:
            print(f"No memories found for user '{args.user}'")
            return

        print(f"Found {len(results['ids'])} memories for user '{args.user}'")
        if not args.force:
            confirm = input("Delete all? [y/N]: ")
            if confirm.lower() != "y":
                print("Cancelled.")
                return

        collection.delete(ids=results["ids"])
        print(f"Deleted {len(results['ids'])} memories")

    elif args.content_contains:
        # Delete memories containing specific text
        results = collection.get(include=["documents"])

        to_delete = []
        for i, doc in enumerate(results["documents"]):
            if args.content_contains.lower() in doc.lower():
                to_delete.append(results["ids"][i])

        if not to_delete:
            print(f"No memories containing '{args.content_contains}'")
            return

        print(f"Found {len(to_delete)} memories containing '{args.content_contains}':")
        for mem_id in to_delete[:10]:  # Show first 10
            result = collection.get(ids=[mem_id], include=["documents"])
            print(f"  - {result['documents'][0][:60]}...")
        if len(to_delete) > 10:
            print(f"  ... and {len(to_delete) - 10} more")

        if not args.force:
            confirm = input("Delete all? [y/N]: ")
            if confirm.lower() != "y":
                print("Cancelled.")
                return

        collection.delete(ids=to_delete)
        print(f"Deleted {len(to_delete)} memories")

    else:
        print("Please specify --id, --user, or --content-contains")


def cmd_stats(args):
    """Show database statistics."""
    collection = get_collection(args.chroma_path)

    total = collection.count()
    print(f"\nTotal memories: {total}")

    if total == 0:
        return

    # Get all to compute stats
    results = collection.get(include=["metadatas"])

    # Count by user
    users = {}
    types = {}
    intensities = []

    for meta in results["metadatas"]:
        user = meta.get("user", "") or "(global)"
        users[user] = users.get(user, 0) + 1

        mtype = meta.get("type", "unknown")
        types[mtype] = types.get(mtype, 0) + 1

        intensities.append(meta.get("intensity", 0.5))

    print(f"\nBy user:")
    for user, count in sorted(users.items(), key=lambda x: -x[1]):
        print(f"  {user}: {count}")

    print(f"\nBy type:")
    for mtype, count in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {mtype}: {count}")

    print(f"\nIntensity stats:")
    print(f"  Min: {min(intensities):.2f}")
    print(f"  Max: {max(intensities):.2f}")
    print(f"  Avg: {sum(intensities)/len(intensities):.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Manage ChromaDB memories for bicker-bot"
    )
    parser.add_argument(
        "--chroma-path",
        default="./data/chroma",
        help="Path to ChromaDB storage (default: ./data/chroma)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # list command
    list_parser = subparsers.add_parser("list", help="List memories")
    list_parser.add_argument("--user", help="Filter by user")
    list_parser.add_argument("--type", help="Filter by type (fact, opinion, interaction, event)")
    list_parser.add_argument("--min-intensity", type=float, help="Minimum intensity")
    list_parser.add_argument("--max-intensity", type=float, help="Maximum intensity")
    list_parser.add_argument("--limit", type=int, default=50, help="Max results (default: 50)")
    list_parser.set_defaults(func=cmd_list)

    # search command
    search_parser = subparsers.add_parser("search", help="Search memories by query")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    search_parser.set_defaults(func=cmd_search)

    # show command
    show_parser = subparsers.add_parser("show", help="Show full details of a memory")
    show_parser.add_argument("id", help="Memory ID (can be partial)")
    show_parser.set_defaults(func=cmd_show)

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete memories")
    delete_parser.add_argument("--id", help="Delete by ID")
    delete_parser.add_argument("--user", help="Delete all memories for a user")
    delete_parser.add_argument("--content-contains", help="Delete memories containing text")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    delete_parser.set_defaults(func=cmd_delete)

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
