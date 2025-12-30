"""SQLite storage for conversation history."""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from bicker_bot.core.router import TimestampedMessage
from bicker_bot.irc.client import Message, MessageType

logger = logging.getLogger(__name__)


class ConversationStore:
    """SQLite-backed conversation storage."""

    def __init__(self, db_path: Path | str, max_per_channel: int = 1000):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_per_channel = max_per_channel
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()
        logger.info(f"ConversationStore initialized at {self.db_path}")

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel TEXT NOT NULL,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                message_type TEXT NOT NULL DEFAULT 'NORMAL',
                timestamp TIMESTAMP NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_channel_time
                ON messages(channel, timestamp DESC);
        """)
        self._conn.commit()

    def save_message(
        self,
        channel: str,
        sender: str,
        content: str,
        message_type: MessageType,
        timestamp: datetime,
    ) -> None:
        """Save a message and cleanup old ones."""
        self._conn.execute(
            """
            INSERT INTO messages (channel, sender, content, message_type, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (channel, sender, content, message_type.name, timestamp.isoformat()),
        )

        # Cleanup: delete messages beyond max_per_channel
        self._conn.execute(
            """
            DELETE FROM messages
            WHERE channel = ? AND id NOT IN (
                SELECT id FROM messages
                WHERE channel = ?
                ORDER BY timestamp DESC
                LIMIT ?
            )
            """,
            (channel, channel, self._max_per_channel),
        )
        self._conn.commit()

    def load_recent(self, channel: str, limit: int = 30) -> list[TimestampedMessage]:
        """Load recent messages for a channel.

        Returns messages oldest-first (chronological order).
        """
        rows = self._conn.execute(
            """
            SELECT sender, content, message_type, timestamp
            FROM messages
            WHERE channel = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (channel, limit),
        ).fetchall()

        messages = []
        for row in reversed(rows):  # Reverse to get oldest-first
            msg = Message(
                channel=channel,
                sender=row["sender"],
                content=row["content"],
                type=MessageType[row["message_type"]],
            )
            messages.append(TimestampedMessage(
                message=msg,
                timestamp=datetime.fromisoformat(row["timestamp"]),
            ))

        return messages

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
