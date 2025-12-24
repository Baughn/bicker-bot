"""Bot personality definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .hachiman import get_hachiman_prompt
from .merry import get_merry_prompt

if TYPE_CHECKING:
    from bicker_bot.config import Config


def get_personality_prompt(bot: str, config: Config) -> str:
    """Get the personality prompt for the given bot with configured nickname.

    Args:
        bot: Which bot to get the prompt for ("merry" or "hachiman")
        config: Application configuration (for nickname lookup)

    Returns:
        The full personality prompt with the correct nickname
    """
    if bot == "merry":
        return get_merry_prompt(config.irc.nick_merry)
    else:
        return get_hachiman_prompt(config.irc.nick_hachiman)


__all__ = ["get_hachiman_prompt", "get_merry_prompt", "get_personality_prompt"]
