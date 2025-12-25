"""Logging utilities for bicker-bot."""

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Generator

# AI debug mode flag
_ai_debug: bool = False
_ai_debug_lock = Lock()


def set_ai_debug(enabled: bool) -> None:
    """Enable or disable AI debug logging."""
    global _ai_debug
    with _ai_debug_lock:
        _ai_debug = enabled


def is_ai_debug() -> bool:
    """Check if AI debug logging is enabled."""
    with _ai_debug_lock:
        return _ai_debug


# Dedicated logger for AI debug output
_ai_logger = logging.getLogger("bicker_bot.ai_debug")


def log_llm_call(
    operation: str,
    model: str,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    tools: list[Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Log the input to an LLM call when AI debug is enabled."""
    if not is_ai_debug():
        return

    parts = [
        f"\n{'='*80}",
        f"LLM CALL: {operation}",
        f"Model: {model}",
        f"{'='*80}",
    ]

    if system_prompt:
        parts.append(f"\n--- SYSTEM PROMPT ---\n{system_prompt}")

    if user_prompt:
        parts.append(f"\n--- USER PROMPT ---\n{user_prompt}")

    if messages:
        parts.append(f"\n--- MESSAGES ---\n{json.dumps(messages, indent=2, default=str)}")

    if tools:
        parts.append(f"\n--- TOOLS ---\n{_format_tools(tools)}")

    if config:
        parts.append(f"\n--- CONFIG ---\n{json.dumps(config, indent=2, default=str)}")

    _ai_logger.info("\n".join(parts))


def log_llm_response(
    operation: str,
    response_text: str | None = None,
    tool_calls: list[Any] | None = None,
    usage: dict[str, Any] | None = None,
    raw_response: Any = None,
) -> None:
    """Log the output from an LLM call when AI debug is enabled."""
    if not is_ai_debug():
        return

    parts = [
        f"\n{'-'*80}",
        f"LLM RESPONSE: {operation}",
        f"{'-'*80}",
    ]

    if response_text:
        parts.append(f"\n--- RESPONSE TEXT ---\n{response_text}")

    if tool_calls:
        parts.append(f"\n--- TOOL CALLS ---\n{json.dumps(tool_calls, indent=2, default=str)}")

    if usage:
        parts.append(f"\n--- USAGE ---\n{json.dumps(usage, indent=2, default=str)}")

    if raw_response and not response_text and not tool_calls:
        parts.append(f"\n--- RAW RESPONSE ---\n{raw_response}")

    parts.append(f"{'='*80}\n")

    _ai_logger.info("\n".join(parts))


def log_rag_query(
    operation: str,
    query: str,
    filters: dict[str, Any] | None = None,
    limit: int | None = None,
) -> None:
    """Log a RAG query when AI debug is enabled."""
    if not is_ai_debug():
        return

    parts = [
        f"\n{'='*80}",
        f"RAG QUERY: {operation}",
        f"{'='*80}",
        f"\n--- QUERY ---\n{query}",
    ]

    if filters:
        parts.append(f"\n--- FILTERS ---\n{json.dumps(filters, indent=2, default=str)}")

    if limit is not None:
        parts.append(f"\n--- LIMIT ---\n{limit}")

    _ai_logger.info("\n".join(parts))


def log_rag_results(
    operation: str,
    results: list[Any],
    distances: list[float] | None = None,
) -> None:
    """Log RAG results when AI debug is enabled."""
    if not is_ai_debug():
        return

    parts = [
        f"\n{'-'*80}",
        f"RAG RESULTS: {operation} ({len(results)} results)",
        f"{'-'*80}",
    ]

    for i, result in enumerate(results):
        distance_str = f" (distance: {distances[i]:.4f})" if distances and i < len(distances) else ""
        if hasattr(result, "content"):
            parts.append(f"\n[{i+1}]{distance_str}\n{result.content}")
        elif isinstance(result, dict):
            parts.append(f"\n[{i+1}]{distance_str}\n{json.dumps(result, indent=2, default=str)}")
        else:
            parts.append(f"\n[{i+1}]{distance_str}\n{result}")

    parts.append(f"{'='*80}\n")

    _ai_logger.info("\n".join(parts))


def _format_tools(tools: list[Any]) -> str:
    """Format tools for logging."""
    result = []
    for tool in tools:
        if hasattr(tool, "function_declarations"):
            for func in tool.function_declarations:
                result.append(f"- {func.name}: {func.description}")
        elif hasattr(tool, "name"):
            result.append(f"- {tool.name}")
        else:
            result.append(f"- {tool}")
    return "\n".join(result) if result else str(tools)


@dataclass
class SessionStats:
    """Cumulative statistics for a session.

    Thread-safe counters for tracking bot activity metrics.
    """

    messages_received: int = 0
    gate_passes: int = 0
    gate_fails: int = 0
    engagement_passes: int = 0
    engagement_fails: int = 0
    responses_merry: int = 0
    responses_hachiman: int = 0
    memories_searched: int = 0
    memories_found: int = 0
    memories_stored: int = 0
    api_calls: dict[str, int] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def increment(self, stat: str, amount: int = 1) -> None:
        """Increment a stat counter."""
        with self._lock:
            if hasattr(self, stat) and stat != "_lock":
                current = getattr(self, stat)
                if isinstance(current, int):
                    setattr(self, stat, current + amount)

    def increment_api_call(self, model: str) -> None:
        """Track an API call to a specific model."""
        with self._lock:
            self.api_calls[model] = self.api_calls.get(model, 0) + 1

    def summary(self) -> dict[str, Any]:
        """Return a summary of all stats."""
        with self._lock:
            total_gate = self.gate_passes + self.gate_fails
            total_engage = self.engagement_passes + self.engagement_fails

            return {
                "received": self.messages_received,
                "gate_rate": f"{100 * self.gate_passes / max(1, total_gate):.0f}%",
                "engage_rate": f"{100 * self.engagement_passes / max(1, total_engage):.0f}%",
                "merry": self.responses_merry,
                "hachi": self.responses_hachiman,
                "memories_stored": self.memories_stored,
                "api_calls": dict(self.api_calls),
            }

    def summary_line(self) -> str:
        """Return a single-line summary for logging."""
        with self._lock:
            total_gate = self.gate_passes + self.gate_fails
            total_engage = self.engagement_passes + self.engagement_fails
            gate_pct = 100 * self.gate_passes / max(1, total_gate)
            engage_pct = 100 * self.engagement_passes / max(1, total_engage)

            return (
                f"received={self.messages_received} "
                f"gate_rate={gate_pct:.0f}% engage_rate={engage_pct:.0f}% "
                f"merry={self.responses_merry} hachi={self.responses_hachiman} "
                f"memories_stored={self.memories_stored}"
            )


# Global session stats instance
_session_stats: SessionStats | None = None
_stats_lock = Lock()


def get_session_stats() -> SessionStats:
    """Get the global session stats instance."""
    global _session_stats
    with _stats_lock:
        if _session_stats is None:
            _session_stats = SessionStats()
        return _session_stats


def reset_session_stats() -> None:
    """Reset session stats (mainly for testing)."""
    global _session_stats
    with _stats_lock:
        _session_stats = SessionStats()


# Dedicated logger for LLM round summaries (always on)
_llm_logger = logging.getLogger("bicker_bot.llm")


def log_llm_round(
    component: str,
    model: str,
    round_num: int,
    tokens_in: int | None,
    tokens_out: int | None,
    tools_called: list[str] | None = None,
    stop_reason: str | None = None,
) -> None:
    """Log a summary of an LLM round (always on).

    Args:
        component: Which component made the call (e.g., "context", "responder/hachiman")
        model: Model name used
        round_num: Round number in multi-turn tool calling (0-indexed)
        tokens_in: Input token count (None if unavailable)
        tokens_out: Output token count (None if unavailable)
        tools_called: List of tool names called in this round
        stop_reason: Stop reason (for Claude: end_turn, tool_use, etc.)
    """
    tools_str = f" tools={tools_called}" if tools_called else ""
    tokens_str = f"in={tokens_in or '?'} out={tokens_out or '?'}"
    stop_str = f" stop={stop_reason}" if stop_reason else ""

    _llm_logger.info(
        f"LLM_ROUND [{component}] model={model} round={round_num} "
        f"{tokens_str}{tools_str}{stop_str}"
    )


@contextmanager
def log_timing(
    logger: logging.Logger, operation: str
) -> Generator[None, None, None]:
    """Context manager for timing operations.

    Logs at DEBUG level on completion.

    Example:
        with log_timing(logger, "Gate check"):
            result = gate.should_respond(...)
        # Logs: "Gate check completed in 1.23ms"
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"{operation} completed in {elapsed_ms:.2f}ms")
