"""Main entry point for bicker-bot."""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from bicker_bot.config import Config, load_config
from bicker_bot.orchestrator import Orchestrator


def setup_logging(debug: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


def load_api_keys_from_env(config: Config) -> Config:
    """Load API keys from environment if not in config."""
    # This is a workaround since pydantic-settings doesn't auto-load nested secrets
    if not config.llm.anthropic_api_key:
        key = os.getenv("ANTHROPIC_API_KEY")
        if key:
            from pydantic import SecretStr

            config.llm.anthropic_api_key = SecretStr(key)

    if not config.llm.google_api_key:
        # Support both GOOGLE_API_KEY and GEMINI_API_KEY
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if key:
            from pydantic import SecretStr

            config.llm.google_api_key = SecretStr(key)

    return config


async def async_main(
    config_path: str | None = None,
    debug: bool = False,
    debug_ai: bool = False,
    deduplicate_memories: bool = False,
    dedup_dry_run: bool = False,
) -> None:
    """Async main entry point."""
    setup_logging(debug)
    logger = logging.getLogger(__name__)

    # Load .env file if present
    load_dotenv()

    # Load configuration
    config = load_config(config_path)
    config = load_api_keys_from_env(config)

    # Set global AI debug flag
    if debug_ai:
        from bicker_bot.core.logging import set_ai_debug
        set_ai_debug(True)
        logger.info("AI debug logging enabled - full LLM/RAG inputs and outputs will be logged")

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
            f"checked={report.memories_scanned}, "
            f"clusters={report.clusters_found}, "
            f"merged={report.memories_merged}, "
            f"deleted={report.memories_deleted}"
        )

        if dedup_dry_run:
            logger.info("Dry run complete, exiting without starting bot")
            return

    logger.info("Starting Bicker-Bot...")
    logger.info(f"Merry nick: {config.irc.nick_merry}")
    logger.info(f"Hachiman nick: {config.irc.nick_hachiman}")
    logger.info(f"Channels: {', '.join(config.irc.channels)}")

    # Create and start orchestrator
    orchestrator = Orchestrator(config)

    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point (sync wrapper)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Bicker-Bot: Sibling AI bots for IRC",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--debug-ai",
        action="store_true",
        help="Log full inputs and outputs for all LLM and RAG calls",
    )
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

    args = parser.parse_args()

    asyncio.run(async_main(
        config_path=args.config,
        debug=args.debug,
        debug_ai=args.debug_ai,
        deduplicate_memories=args.deduplicate_memories,
        dedup_dry_run=args.dedup_dry_run,
    ))


if __name__ == "__main__":
    main()
