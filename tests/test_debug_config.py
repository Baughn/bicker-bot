"""Tests for debug config loader."""

from pathlib import Path

import pytest

from bicker_bot.debug.config_loader import ConfigLoader, PromptConfig


class TestConfigLoader:
    """Tests for ConfigLoader."""

    @pytest.fixture
    def config_dir(self, tmp_path: Path) -> Path:
        """Create a config directory with test files."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        # Create a test prompt file
        (prompts_dir / "merry.md").write_text("""---
model: gemini-3-pro-preview
max_tokens: 1024
temperature: 0.9
---

You are Merry Nightmare, a dream demon.
Direct and action-oriented.
""")

        # Create policies file
        (tmp_path / "policies.yaml").write_text("""
gate:
  base_prob: 0.05
  decay_factor: 0.5

context:
  max_tool_rounds: 3
""")

        return tmp_path

    def test_load_prompt(self, config_dir: Path):
        """Test loading a prompt file."""
        loader = ConfigLoader(config_dir)
        prompt = loader.get_prompt("merry")

        assert prompt is not None
        assert isinstance(prompt, PromptConfig)
        assert prompt.model == "gemini-3-pro-preview"
        assert prompt.max_tokens == 1024
        assert prompt.temperature == 0.9
        assert "Merry Nightmare" in prompt.content

    def test_load_policies(self, config_dir: Path):
        """Test loading policies file."""
        loader = ConfigLoader(config_dir)
        policies = loader.get_policies()

        assert policies["gate"]["base_prob"] == 0.05
        assert policies["context"]["max_tool_rounds"] == 3

    def test_snapshot(self, config_dir: Path):
        """Test getting a frozen snapshot."""
        loader = ConfigLoader(config_dir)
        snapshot1 = loader.snapshot()
        snapshot2 = loader.snapshot()

        # Should be equal but not the same object
        assert snapshot1 == snapshot2
        assert snapshot1 is not snapshot2

    def test_reload(self, config_dir: Path):
        """Test hot-reloading config."""
        loader = ConfigLoader(config_dir)
        original = loader.get_policies()

        # Modify the file
        (config_dir / "policies.yaml").write_text("""
gate:
  base_prob: 0.10
""")

        # Reload
        loader.reload()
        updated = loader.get_policies()

        assert original["gate"]["base_prob"] == 0.05
        assert updated["gate"]["base_prob"] == 0.10

    def test_missing_prompt(self, config_dir: Path):
        """Test getting a nonexistent prompt."""
        loader = ConfigLoader(config_dir)
        prompt = loader.get_prompt("nonexistent")
        assert prompt is None

    def test_list_prompts(self, config_dir: Path):
        """Test listing available prompts."""
        loader = ConfigLoader(config_dir)
        prompts = loader.list_prompts()
        assert "merry" in prompts
