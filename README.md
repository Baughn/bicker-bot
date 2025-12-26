# Bicker-Bot

Two AI bots (Merry & Hachiman) that bicker like siblings on IRC.

## Features

- **Merry Nightmare** (Gemini 3 Pro) - Dream demon, direct and action-oriented
- **Hachiman Hikigaya** (Claude Opus 4.5) - Cynical observer with hidden depth
- Statistical response gating to avoid overwhelming humans
- RAG-based memory system for per-user context
- ChromaDB with local GPU embeddings

## Setup

```bash
# Enter development shell
nix develop

# Copy and configure
cp config.yaml.example config.yaml

# Run
bicker-bot
```

## Development

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=bicker_bot
```
