"""Configuration loading and validation."""

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class IRCConfig(BaseModel):
    """IRC connection configuration."""

    server: str = "irc.libera.chat"
    port: int = 6697
    ssl: bool = True
    nick_merry: str = "Merry"
    nick_hachiman: str = "Hachi"
    channels: list[str] = Field(default_factory=lambda: ["#bicker-bot"])
    nickserv_password_merry: SecretStr | None = None
    nickserv_password_hachiman: SecretStr | None = None


class LLMConfig(BaseModel):
    """LLM API configuration."""

    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None


class MemoryConfig(BaseModel):
    """Memory/RAG configuration."""

    chroma_path: Path = Path("./data/chroma")
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    high_intensity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7

    # Deduplication settings
    dedup_enabled: bool = True
    dedup_upper_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.95
    dedup_lower_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.90


class GateConfig(BaseModel):
    """Response gate probability configuration."""

    base_prob: Annotated[float, Field(ge=0.0, le=1.0)] = 0.05
    mention_prob: Annotated[float, Field(ge=0.0, le=1.0)] = 0.8
    question_prob: Annotated[float, Field(ge=0.0, le=1.0)] = 0.3
    conversation_start_prob: Annotated[float, Field(ge=0.0, le=1.0)] = 0.4
    mode_change_prob: Annotated[float, Field(ge=0.0, le=1.0)] = 0.9
    decay_factor: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    silence_threshold_minutes: Annotated[int, Field(ge=1)] = 5


class Config(BaseSettings):
    """Main application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="BICKER_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    irc: IRCConfig = Field(default_factory=IRCConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    gate: GateConfig = Field(default_factory=GateConfig)


def load_config(config_path: Path | str | None = None) -> Config:
    """Load configuration from YAML file and environment variables.

    Priority (highest to lowest):
    1. Environment variables (BICKER_* prefix)
    2. YAML config file
    3. Default values

    Args:
        config_path: Path to YAML config file. If None, tries ./config.yaml

    Returns:
        Validated configuration object
    """
    if config_path is None:
        config_path = Path("config.yaml")
    else:
        config_path = Path(config_path)

    yaml_config: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f) or {}

    # Filter out None values from YAML (e.g., "llm:" with no values parses as None)
    yaml_config = {k: v for k, v in yaml_config.items() if v is not None}

    # Merge YAML config into environment-based config
    # pydantic-settings handles env vars automatically
    return Config(**yaml_config)


def get_bot_nicks(config: Config) -> tuple[str, str]:
    """Get both bot nicknames for mention detection."""
    return (config.irc.nick_merry, config.irc.nick_hachiman)
