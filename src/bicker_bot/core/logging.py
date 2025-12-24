"""Logging utilities for bicker-bot."""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Generator


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
