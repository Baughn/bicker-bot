"""SQLite storage for traces."""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bicker_bot.tracing.context import TraceContext

logger = logging.getLogger(__name__)


@dataclass
class TraceSummary:
    """Lightweight summary of a trace for list views."""

    id: str
    created_at: datetime
    channel: str
    bot: str | None
    trigger_text: str
    outcome: str
    is_replay: bool


class TraceStore:
    """SQLite-backed trace storage."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()
        logger.info(f"TraceStore initialized at {self.db_path}")

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS traces (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                channel TEXT NOT NULL,
                bot TEXT,
                trigger_text TEXT,
                outcome TEXT NOT NULL,
                is_replay INTEGER DEFAULT 0,
                original_trace_id TEXT,
                trace_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_traces_channel ON traces(channel);
            CREATE INDEX IF NOT EXISTS idx_traces_bot ON traces(bot);
        """)
        self._conn.commit()

    def save(self, trace: TraceContext) -> None:
        """Save a trace to the database."""
        # Determine outcome based on steps and final_result
        if trace.final_result:
            outcome = "responded"
        elif any(s.stage == "engagement" for s in trace.steps):
            outcome = "declined_engagement"
        else:
            outcome = "declined_gate"

        # Determine which bot responded (if any)
        bot = None
        for step in trace.steps:
            if step.stage == "selector" and "selected" in step.outputs:
                bot = step.outputs["selected"]
                break
            if step.stage == "responder" and "bot" in step.inputs:
                # Get bot from responder inputs (most reliable)
                bot = step.inputs["bot"]
                break

        # Get first trigger message for preview
        trigger_text = trace.trigger_messages[0][:100] if trace.trigger_messages else ""

        self._conn.execute(
            """
            INSERT OR REPLACE INTO traces
            (id, created_at, channel, bot, trigger_text, outcome, is_replay, original_trace_id, trace_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.id,
                trace.started_at.isoformat(),
                trace.channel,
                bot,
                trigger_text,
                outcome,
                1 if trace.is_replay else 0,
                trace.original_trace_id,
                json.dumps(trace.to_dict()),
            ),
        )
        self._conn.commit()
        logger.debug(f"Saved trace {trace.id[:8]}... outcome={outcome}")

    def get(self, trace_id: str) -> TraceContext | None:
        """Get a trace by ID."""
        row = self._conn.execute(
            "SELECT trace_json FROM traces WHERE id = ?", (trace_id,)
        ).fetchone()
        if row is None:
            return None
        return TraceContext.from_dict(json.loads(row["trace_json"]))

    def recent(
        self,
        limit: int = 50,
        channel: str | None = None,
        bot: str | None = None,
    ) -> list[TraceSummary]:
        """Get recent trace summaries."""
        query = "SELECT id, created_at, channel, bot, trigger_text, outcome, is_replay FROM traces"
        params: list = []
        conditions = []

        if channel:
            conditions.append("channel = ?")
            params.append(channel)
        if bot:
            conditions.append("bot = ?")
            params.append(bot)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [
            TraceSummary(
                id=row["id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                channel=row["channel"],
                bot=row["bot"],
                trigger_text=row["trigger_text"] or "",
                outcome=row["outcome"],
                is_replay=bool(row["is_replay"]),
            )
            for row in rows
        ]

    def prune(self, keep_last: int = 500) -> int:
        """Delete old traces, keeping the most recent. Returns count deleted."""
        # Get the ID of the Nth most recent trace
        cutoff_row = self._conn.execute(
            "SELECT id FROM traces ORDER BY created_at DESC LIMIT 1 OFFSET ?",
            (keep_last - 1,),
        ).fetchone()

        if cutoff_row is None:
            return 0  # Fewer traces than keep_last

        # Get the created_at of that trace
        cutoff = self._conn.execute(
            "SELECT created_at FROM traces WHERE id = ?", (cutoff_row["id"],)
        ).fetchone()

        if cutoff is None:
            return 0

        # Delete older traces
        result = self._conn.execute(
            "DELETE FROM traces WHERE created_at < ?", (cutoff["created_at"],)
        )
        self._conn.commit()
        deleted = result.rowcount
        if deleted > 0:
            logger.info(f"Pruned {deleted} old traces")
        return deleted

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
