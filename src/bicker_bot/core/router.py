"""Message routing and buffering."""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bicker_bot.irc.client import Message

logger = logging.getLogger(__name__)


@dataclass
class TimestampedMessage:
    """Message with timestamp for buffer management."""

    message: "Message"
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def channel(self) -> str:
        return self.message.channel

    @property
    def sender(self) -> str:
        return self.message.sender

    @property
    def content(self) -> str:
        return self.message.content

    @property
    def is_action(self) -> bool:
        return self.message.is_action


@dataclass
class ChannelBuffer:
    """Rolling buffer of messages for a channel."""

    max_size: int = 30
    _messages: deque[TimestampedMessage] = field(default_factory=deque)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add(self, message: "Message") -> None:
        """Add a message to the buffer."""
        async with self._lock:
            ts_msg = TimestampedMessage(message=message)
            self._messages.append(ts_msg)
            # Trim to max size
            while len(self._messages) > self.max_size:
                self._messages.popleft()

    async def get_recent(self, count: int | None = None) -> list[TimestampedMessage]:
        """Get recent messages from buffer.

        Args:
            count: Number of messages to get. None = all (up to max_size)

        Returns:
            List of messages, oldest first
        """
        async with self._lock:
            if count is None:
                return list(self._messages)
            return list(self._messages)[-count:]

    async def get_last_message_time(self) -> datetime | None:
        """Get timestamp of most recent message."""
        async with self._lock:
            if self._messages:
                return self._messages[-1].timestamp
            return None

    async def count_consecutive_bot_messages(self, bot_nicks: tuple[str, str]) -> int:
        """Count consecutive bot messages at the end of the buffer.

        Used for bickering decay calculation.
        """
        async with self._lock:
            count = 0
            for msg in reversed(self._messages):
                if msg.sender in bot_nicks:
                    count += 1
                else:
                    break
            return count

    def __len__(self) -> int:
        return len(self._messages)


class MessageRouter:
    """Routes messages and maintains per-channel buffers."""

    def __init__(self, buffer_size: int = 30):
        self._buffer_size = buffer_size
        self._buffers: dict[str, ChannelBuffer] = {}
        self._lock = asyncio.Lock()

    async def _get_buffer(self, channel: str) -> ChannelBuffer:
        """Get or create buffer for a channel."""
        async with self._lock:
            if channel not in self._buffers:
                self._buffers[channel] = ChannelBuffer(max_size=self._buffer_size)
            return self._buffers[channel]

    async def add_message(self, message: "Message") -> None:
        """Add a message to the appropriate channel buffer."""
        buffer = await self._get_buffer(message.channel)
        await buffer.add(message)
        logger.debug(f"Added message to {message.channel} buffer (size: {len(buffer)})")

    async def get_recent_messages(
        self, channel: str, count: int | None = None
    ) -> list[TimestampedMessage]:
        """Get recent messages for a channel."""
        buffer = await self._get_buffer(channel)
        return await buffer.get_recent(count)

    async def get_last_activity(self, channel: str) -> datetime | None:
        """Get timestamp of last activity in channel."""
        buffer = await self._get_buffer(channel)
        return await buffer.get_last_message_time()

    async def count_consecutive_bot_messages(
        self, channel: str, bot_nicks: tuple[str, str]
    ) -> int:
        """Count consecutive bot messages for decay calculation."""
        buffer = await self._get_buffer(channel)
        return await buffer.count_consecutive_bot_messages(bot_nicks)

    def format_context(self, messages: list[TimestampedMessage]) -> str:
        """Format messages for LLM context.

        Returns a string like:
        [12:34] <Alice> Hello!
        [12:35] <Bob> Hi there
        [12:36] * Bob waves
        """
        lines = []
        for msg in messages:
            time_str = msg.timestamp.strftime("%H:%M")
            if msg.is_action:
                lines.append(f"[{time_str}] * {msg.sender} {msg.content}")
            else:
                lines.append(f"[{time_str}] <{msg.sender}> {msg.content}")
        return "\n".join(lines)
