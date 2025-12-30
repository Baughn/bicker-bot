"""IRC client for bot connections."""

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import pydle

from bicker_bot.config import Config

logger = logging.getLogger(__name__)

# IRC has a 512 byte limit per message including protocol overhead.
# Account for `:nick!user@host PRIVMSG #channel :` prefix and `\r\n` suffix.
# Using 400 chars as a safe limit for message content.
MAX_MESSAGE_LENGTH = 400
MESSAGE_SPLIT_DELAY = 0.33  # Delay between split message parts


def split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a message into chunks that fit within IRC limits.

    Splits at word boundaries when possible, falling back to hard splits.
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        # Find a good split point (last space within limit)
        split_at = text.rfind(" ", 0, max_length)
        if split_at == -1 or split_at < max_length // 2:
            # No good split point, hard split
            split_at = max_length

        chunks.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()

    return chunks


class MessageType(Enum):
    """Type of IRC message."""

    NORMAL = auto()
    ACTION = auto()  # /me actions
    MODE_CHANGE = auto()  # channel mode changes


@dataclass
class Message:
    """Represents an IRC message."""

    channel: str
    sender: str
    content: str
    type: MessageType = MessageType.NORMAL

    def __str__(self) -> str:
        if self.type == MessageType.ACTION:
            return f"* {self.sender} {self.content}"
        return f"<{self.sender}> {self.content}"


MessageHandler = Callable[[Message], Coroutine[Any, Any, None]]


class BotClient(pydle.Client):
    """Single bot IRC client."""

    def __init__(
        self,
        nickname: str,
        channels: list[str],
        on_message: MessageHandler | None = None,
        nickserv_password: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(nickname, **kwargs)
        self._channels_to_join = channels
        self._on_message = on_message
        self._nickserv_password = nickserv_password
        self._ready = asyncio.Event()
        self._connect_time: float = 0.0
        self._recent_joins: dict[str, float] = {}  # "channel#nick" -> join time

    async def on_connect(self) -> None:
        """Handle successful connection."""
        logger.info(f"[{self.nickname}] Connected to server")
        self._connect_time = time.monotonic()

        # Identify with NickServ if password provided
        if self._nickserv_password:
            await self.message("NickServ", f"IDENTIFY {self._nickserv_password}")
            # Give NickServ a moment
            await asyncio.sleep(1)

        # Join configured channels
        for channel in self._channels_to_join:
            await self.join(channel)
            logger.info(f"[{self.nickname}] Joined {channel}")

        self._ready.set()

    async def on_join(self, channel: str, user: str) -> None:
        """Track user joins for filtering auto-op mode changes."""
        key = f"{channel.lower()}#{user.lower()}"
        self._recent_joins[key] = time.monotonic()

    async def on_part(self, channel: str, user: str, message: str | None = None) -> None:
        """Track user parts to reset join state for mode filtering."""
        key = f"{channel.lower()}#{user.lower()}"
        self._recent_joins.pop(key, None)

    async def on_kick(self, channel: str, target: str, by: str, reason: str | None = None) -> None:
        """Remove kicked user from join tracking."""
        key = f"{channel.lower()}#{target.lower()}"
        self._recent_joins.pop(key, None)

    async def on_quit(self, user: str, message: str | None = None) -> None:
        """Remove user from all channel join tracking when they quit."""
        user_lower = user.lower()
        # Remove all entries for this user across all channels
        keys_to_remove = [k for k in self._recent_joins if k.endswith(f"#{user_lower}")]
        for key in keys_to_remove:
            del self._recent_joins[key]

    async def on_message(self, target: str, source: str, message: str) -> None:
        """Handle incoming channel/private messages."""
        # Ignore our own messages
        if source == self.nickname:
            return

        # Only process channel messages (not PMs)
        if not target.startswith("#"):
            return

        msg = Message(channel=target, sender=source, content=message)
        logger.debug(f"[{self.nickname}] Received: {msg}")

        if self._on_message:
            try:
                await self._on_message(msg)
            except Exception:
                logger.exception(f"[{self.nickname}] Error handling message")

    async def on_ctcp_action(self, source: str, target: str, message: str) -> None:
        """Handle /me actions.

        Note: pydle's on_ctcp_action signature is (by, target, contents) which differs
        from on_message's (target, source, message). The first two params are swapped.
        """
        if source == self.nickname:
            return

        if not target.startswith("#"):
            return

        msg = Message(channel=target, sender=source, content=message, type=MessageType.ACTION)
        logger.debug(f"[{self.nickname}] Received action: {msg}")

        if self._on_message:
            try:
                await self._on_message(msg)
            except Exception:
                logger.exception(f"[{self.nickname}] Error handling action")

    async def on_mode_change(self, channel: str, modes: list[str], by: str) -> None:
        """Handle channel mode changes.

        Args:
            channel: The channel where modes changed
            modes: List of mode strings (e.g., ['+o', 'username'])
            by: Nickname of who changed the mode
        """
        if self._on_message is None:
            return

        # Ignore mode changes in first 10 seconds (initial state sync on join)
        if time.monotonic() - self._connect_time < 10:
            logger.debug(f"[{self.nickname}] Ignoring startup mode change: {modes}")
            return

        # Only process channel modes
        if not channel.startswith("#"):
            return

        # Check if mode target joined recently (auto-op filtering)
        # modes like ['+o', 'username'] - target is after the mode chars
        if len(modes) >= 2:
            target = modes[-1].lower()
            key = f"{channel.lower()}#{target}"
            if key in self._recent_joins:
                if time.monotonic() - self._recent_joins[key] < 10:
                    logger.debug(f"[{self.nickname}] Ignoring auto-op for recent join: {modes}")
                    return

        mode_str = " ".join(modes)
        content = f"Channel mode set to {mode_str} by {by}"
        msg = Message(
            channel=channel,
            sender=by,
            content=content,
            type=MessageType.MODE_CHANGE,
        )
        logger.debug(f"[{self.nickname}] Mode change: {msg}")

        try:
            await self._on_message(msg)
        except Exception:
            logger.exception(f"[{self.nickname}] Error handling mode change")

    async def send_message(self, channel: str, content: str) -> None:
        """Send a message to a channel, splitting if too long."""
        await self._ready.wait()
        chunks = split_message(content)
        for i, chunk in enumerate(chunks):
            if i > 0:
                await asyncio.sleep(MESSAGE_SPLIT_DELAY)
            await self.message(channel, chunk)
            logger.info(f"MSG_SENT: [{self.nickname}] -> {channel}: {chunk}")

    async def send_action(self, channel: str, content: str) -> None:
        """Send an action (/me) to a channel, splitting if too long."""
        await self._ready.wait()
        chunks = split_message(content)
        for i, chunk in enumerate(chunks):
            if i > 0:
                await asyncio.sleep(MESSAGE_SPLIT_DELAY)
            await self.ctcp(channel, "ACTION", chunk)
            logger.info(f"MSG_SENT: [{self.nickname}] -> {channel}: * {chunk}")

    async def wait_ready(self) -> None:
        """Wait until the client is connected and in channels."""
        await self._ready.wait()


@dataclass
class IRCClient:
    """Manages both bot connections."""

    config: Config
    on_message: MessageHandler | None = None
    _merry: BotClient | None = field(default=None, init=False)
    _hachiman: BotClient | None = field(default=None, init=False)

    async def connect(self) -> None:
        """Connect both bots to IRC."""
        irc_cfg = self.config.irc

        # Get NickServ passwords if set
        merry_pass = (
            irc_cfg.nickserv_password_merry.get_secret_value()
            if irc_cfg.nickserv_password_merry
            else None
        )
        hachi_pass = (
            irc_cfg.nickserv_password_hachiman.get_secret_value()
            if irc_cfg.nickserv_password_hachiman
            else None
        )

        # Create both bot clients
        self._merry = BotClient(
            nickname=irc_cfg.nick_merry,
            channels=irc_cfg.channels,
            on_message=self.on_message,
            nickserv_password=merry_pass,
        )

        self._hachiman = BotClient(
            nickname=irc_cfg.nick_hachiman,
            channels=irc_cfg.channels,
            on_message=None,  # Only Merry reads; Hachiman only sends
            nickserv_password=hachi_pass,
        )

        # Connect clients sequentially with delay to avoid rate limiting
        # (pydle.ClientPool doesn't work with existing event loops)
        logger.info(f"Connecting {irc_cfg.nick_merry}...")
        await self._merry.connect(
            hostname=irc_cfg.server,
            port=irc_cfg.port,
            tls=irc_cfg.ssl,
        )

        await asyncio.sleep(2)  # Delay to avoid connection rate limits

        logger.info(f"Connecting {irc_cfg.nick_hachiman}...")
        await self._hachiman.connect(
            hostname=irc_cfg.server,
            port=irc_cfg.port,
            tls=irc_cfg.ssl,
        )

        # Wait for both to be ready (joined channels)
        await asyncio.gather(
            self._merry.wait_ready(),
            self._hachiman.wait_ready(),
        )

        logger.info("Both bots connected and ready")

    async def run_forever(self) -> None:
        """Run until both bots disconnect."""
        if self._merry is None or self._hachiman is None:
            raise RuntimeError("Not connected. Call connect() first.")
        # pydle already starts handle_forever() as a background task during connect()
        # We just need to wait until the clients disconnect
        while self._merry.connected or self._hachiman.connected:
            await asyncio.sleep(1)

    async def send_as_merry(self, channel: str, content: str) -> None:
        """Send a message as Merry."""
        if self._merry is None:
            raise RuntimeError("Not connected")
        await self._merry.send_message(channel, content)

    async def send_as_hachiman(self, channel: str, content: str) -> None:
        """Send a message as Hachiman."""
        if self._hachiman is None:
            raise RuntimeError("Not connected")
        await self._hachiman.send_message(channel, content)

    async def send(
        self, bot: str, channel: str, content: str, msg_type: MessageType = MessageType.NORMAL
    ) -> None:
        """Send a message or action as the specified bot.

        Args:
            bot: Either "merry" or "hachiman"
            channel: Target channel
            content: Message content (for actions, this is the action text without /me)
            msg_type: Type of message to send (NORMAL or ACTION)
        """
        bot_lower = bot.lower()
        if bot_lower == "merry":
            client = self._merry
        elif bot_lower == "hachiman":
            client = self._hachiman
        else:
            raise ValueError(f"Unknown bot: {bot}")

        if client is None:
            raise RuntimeError("Not connected")

        if msg_type == MessageType.ACTION:
            await client.send_action(channel, content)
        else:
            await client.send_message(channel, content)

    @property
    def merry_nick(self) -> str:
        """Get Merry's current nickname."""
        return self.config.irc.nick_merry

    @property
    def hachiman_nick(self) -> str:
        """Get Hachiman's current nickname."""
        return self.config.irc.nick_hachiman

    def is_bot_message(self, sender: str) -> bool:
        """Check if a message was sent by one of our bots."""
        return sender in (self.merry_nick, self.hachiman_nick)

    def get_channel_users(self, channel: str) -> list[str]:
        """Get list of users currently in a channel.

        Uses Merry's client since she's the message reader with full state.
        """
        if self._merry is None:
            return []
        try:
            # pydle stores channel state in self.channels dict
            if channel in self._merry.channels:
                return list(self._merry.channels[channel]["users"])
        except (KeyError, TypeError):
            pass
        return []
