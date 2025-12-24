"""Bot personality definitions."""

from bicker_bot.memory import BotIdentity

from .hachiman import HACHIMAN_PROMPT
from .merry import MERRY_PROMPT

PERSONALITY_PROMPTS = {
    BotIdentity.MERRY: MERRY_PROMPT,
    BotIdentity.HACHIMAN: HACHIMAN_PROMPT,
}

__all__ = ["HACHIMAN_PROMPT", "MERRY_PROMPT", "PERSONALITY_PROMPTS"]
