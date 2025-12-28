"""Configuration loader for debug-time config files."""

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter
import yaml

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """Configuration for a prompt file."""

    name: str
    content: str
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    thinking: bool | None = None
    raw_frontmatter: dict[str, Any] | None = None


class ConfigLoader:
    """Loads and manages debug-time configuration files."""

    def __init__(self, config_dir: Path | str):
        self.config_dir = Path(config_dir)
        self._prompts: dict[str, PromptConfig] = {}
        self._policies: dict[str, Any] = {}
        self._load_all()

    def _load_all(self) -> None:
        """Load all configuration files."""
        self._load_prompts()
        self._load_policies()

    def _load_prompts(self) -> None:
        """Load all prompt files from prompts/ directory."""
        prompts_dir = self.config_dir / "prompts"
        if not prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {prompts_dir}")
            return

        self._prompts.clear()
        for prompt_file in prompts_dir.glob("*.md"):
            try:
                post = frontmatter.load(prompt_file)
                name = prompt_file.stem

                self._prompts[name] = PromptConfig(
                    name=name,
                    content=post.content,
                    model=post.get("model"),
                    max_tokens=post.get("max_tokens"),
                    temperature=post.get("temperature"),
                    thinking=post.get("thinking"),
                    raw_frontmatter=dict(post.metadata),
                )
                logger.debug(f"Loaded prompt: {name}")
            except Exception as e:
                logger.error(f"Failed to load prompt {prompt_file}: {e}")

        logger.info(f"Loaded {len(self._prompts)} prompts")

    def _load_policies(self) -> None:
        """Load policies.yaml file."""
        policies_file = self.config_dir / "policies.yaml"
        if not policies_file.exists():
            logger.warning(f"Policies file not found: {policies_file}")
            self._policies = {}
            return

        try:
            with open(policies_file) as f:
                self._policies = yaml.safe_load(f) or {}
            logger.info(f"Loaded policies with {len(self._policies)} sections")
        except Exception as e:
            logger.error(f"Failed to load policies: {e}")
            self._policies = {}

    def reload(self) -> None:
        """Hot-reload all configuration files."""
        logger.info("Reloading configuration files")
        self._load_all()

    def get_prompt(self, name: str) -> PromptConfig | None:
        """Get a prompt configuration by name."""
        return self._prompts.get(name)

    def get_policies(self) -> dict[str, Any]:
        """Get the policies configuration."""
        return copy.deepcopy(self._policies)

    def list_prompts(self) -> list[str]:
        """List available prompt names."""
        return list(self._prompts.keys())

    def snapshot(self) -> dict[str, Any]:
        """Return a frozen copy of all configuration."""
        return {
            "prompts": {
                name: {
                    "content": p.content,
                    "model": p.model,
                    "max_tokens": p.max_tokens,
                    "temperature": p.temperature,
                    "thinking": p.thinking,
                }
                for name, p in self._prompts.items()
            },
            "policies": copy.deepcopy(self._policies),
        }
