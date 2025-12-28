"""Tests for configuration loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from bicker_bot.config import (
    Config,
    GateConfig,
    IRCConfig,
    LLMConfig,
    MemoryConfig,
    get_bot_nicks,
    load_config,
)


class TestIRCConfig:
    """Tests for IRC configuration."""

    def test_defaults(self):
        """Test default IRC config values."""
        cfg = IRCConfig()
        assert cfg.server == "irc.libera.chat"
        assert cfg.port == 6697
        assert cfg.ssl is True
        assert cfg.nick_merry == "Merry"
        assert cfg.nick_hachiman == "Hachi"
        assert len(cfg.channels) == 1

    def test_custom_values(self):
        """Test custom IRC config values."""
        cfg = IRCConfig(
            server="custom.irc.net",
            port=6667,
            ssl=False,
            nick_merry="CustomMerry",
            channels=["#chan1", "#chan2"],
        )
        assert cfg.server == "custom.irc.net"
        assert cfg.port == 6667
        assert cfg.ssl is False
        assert cfg.nick_merry == "CustomMerry"
        assert cfg.channels == ["#chan1", "#chan2"]


class TestGateConfig:
    """Tests for gate configuration."""

    def test_defaults(self):
        """Test default gate config values."""
        cfg = GateConfig()
        assert cfg.base_prob == 0.05
        assert cfg.mention_prob == 0.8
        assert cfg.question_prob == 0.3
        assert cfg.decay_factor == 0.5

    def test_probability_bounds(self):
        """Test that probabilities must be between 0 and 1."""
        with pytest.raises(ValidationError):
            GateConfig(base_prob=1.5)

        with pytest.raises(ValidationError):
            GateConfig(mention_prob=-0.1)

    def test_silence_threshold_must_be_positive(self):
        """Test that silence threshold must be >= 1."""
        with pytest.raises(ValidationError):
            GateConfig(silence_threshold_minutes=0)


class TestMemoryConfig:
    """Tests for memory configuration."""

    def test_defaults(self):
        """Test default memory config values."""
        cfg = MemoryConfig()
        assert cfg.chroma_path == Path("./data/chroma")
        assert cfg.high_intensity_threshold == 0.7

    def test_intensity_bounds(self):
        """Test intensity threshold bounds."""
        with pytest.raises(ValidationError):
            MemoryConfig(high_intensity_threshold=1.5)

    def test_dedup_defaults(self):
        """Test dedup config has sensible defaults."""
        config = MemoryConfig()
        assert config.dedup_enabled is True
        assert config.dedup_upper_threshold == 0.95
        assert config.dedup_lower_threshold == 0.90

    def test_dedup_validation(self):
        """Test dedup thresholds are validated."""
        # Valid config
        config = MemoryConfig(dedup_upper_threshold=0.95, dedup_lower_threshold=0.90)
        assert config.dedup_upper_threshold == 0.95

        # Invalid: threshold > 1.0
        with pytest.raises(ValidationError):
            MemoryConfig(dedup_upper_threshold=1.5)

        # Invalid: threshold < 0.0
        with pytest.raises(ValidationError):
            MemoryConfig(dedup_lower_threshold=-0.1)


class TestConfig:
    """Tests for main configuration."""

    def test_defaults(self):
        """Test that Config creates valid defaults."""
        cfg = Config()
        assert isinstance(cfg.irc, IRCConfig)
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.memory, MemoryConfig)
        assert isinstance(cfg.gate, GateConfig)

    def test_nested_override(self):
        """Test overriding nested config values."""
        cfg = Config(
            irc=IRCConfig(server="test.net"),
            gate=GateConfig(base_prob=0.2),
        )
        assert cfg.irc.server == "test.net"
        assert cfg.gate.base_prob == 0.2
        # Other values should be defaults
        assert cfg.irc.port == 6697
        assert cfg.gate.mention_prob == 0.8


class TestLoadConfig:
    """Tests for config file loading."""

    def test_load_from_yaml(self, temp_config_file: Path):
        """Test loading config from YAML file."""
        cfg = load_config(temp_config_file)

        assert cfg.irc.server == "test.irc.net"
        assert cfg.irc.nick_merry == "TestMerry"
        assert cfg.irc.nick_hachiman == "TestHachi"
        assert cfg.irc.channels == ["#test"]
        assert cfg.memory.high_intensity_threshold == 0.8
        assert cfg.gate.base_prob == 0.1
        assert cfg.gate.mention_prob == 0.9

    def test_load_nonexistent_uses_defaults(self, tmp_path: Path):
        """Test that missing config file uses defaults."""
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg.irc.server == "irc.libera.chat"

    def test_get_bot_nicks(self):
        """Test getting bot nicknames."""
        cfg = Config(irc=IRCConfig(nick_merry="M", nick_hachiman="H"))
        nicks = get_bot_nicks(cfg)
        assert nicks == ("M", "H")
