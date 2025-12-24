"""Pytest configuration and fixtures."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from bicker_bot.config import Config, GateConfig, IRCConfig, LLMConfig, MemoryConfig


# Load .env for integration tests
load_dotenv()


@pytest.fixture
def default_config() -> Config:
    """Provide a default configuration for testing."""
    return Config()


@pytest.fixture
def gate_config() -> GateConfig:
    """Provide gate config with predictable values for testing."""
    return GateConfig(
        base_prob=0.1,
        mention_prob=0.8,
        question_prob=0.3,
        conversation_start_prob=0.4,
        decay_factor=0.5,
        silence_threshold_minutes=5,
    )


@pytest.fixture
def bot_nicks() -> tuple[str, str]:
    """Standard bot nicknames for testing."""
    return ("Merry", "Hachi")


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary config file."""
    config_content = """
irc:
  server: "test.irc.net"
  port: 6697
  ssl: true
  nick_merry: "TestMerry"
  nick_hachiman: "TestHachi"
  channels:
    - "#test"

llm:
  anthropic_api_key: "test-key"

memory:
  chroma_path: "./test-data"
  embedding_model: "test-model"
  high_intensity_threshold: 0.8

gate:
  base_prob: 0.1
  mention_prob: 0.9
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def has_api_keys() -> bool:
    """Check if API keys are available for integration tests."""
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return bool(os.getenv("ANTHROPIC_API_KEY") and google_key)
