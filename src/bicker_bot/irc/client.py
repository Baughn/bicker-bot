"""IRC client for bot connections."""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

import pydle

from bicker_bot.config import Config

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents an IRC message."""

    channel: str
    sender: str
    content: str
    is_action: bool = False

    def __str__(self) -> str:
        if self.is_action:
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

    async def on_connect(self) -> None:
        """Handle successful connection."""
        logger.info(f"[{self.nickname}] Connected to server")

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

        msg = Message(channel=target, sender=source, content=message, is_action=True)
        logger.debug(f"[{self.nickname}] Received action: {msg}")

        if self._on_message:
            try:
                await self._on_message(msg)
            except Exception:
                logger.exception(f"[{self.nickname}] Error handling action")

    async def send_message(self, channel: str, content: str) -> None:
        """Send a message to a channel."""
        await self._ready.wait()
        await self.message(channel, content)
        logger.info(f"MSG_SENT: [{self.nickname}] -> {channel}: {content}")

    async def send_action(self, channel: str, content: str) -> None:
        """Send an action (/me) to a channel."""
        await self._ready.wait()
        await self.ctcp(channel, "ACTION", content)
        logger.info(f"MSG_SENT: [{self.nickname}] -> {channel}: * {content}")

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
        self, bot: str, channel: str, content: str, is_action: bool = False
    ) -> None:
        """Send a message or action as the specified bot.

        Args:
            bot: Either "merry" or "hachiman"
            channel: Target channel
            content: Message content (for actions, this is the action text without /me)
            is_action: If True, send as CTCP ACTION (/me)
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

        if is_action:
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
