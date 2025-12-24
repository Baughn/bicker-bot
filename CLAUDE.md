# Bicker-Bot Architecture

Two AI bots (Merry Nightmare / Gemini 3 Pro Preview and Hachiman Hikigaya / Claude Opus 4.5) that bicker like siblings on IRC.

## High-Level Architecture

```
                                    ┌─────────────────────────────────────────┐
                                    │           ORCHESTRATOR                  │
                                    │         (orchestrator.py)               │
                                    └─────────────────────────────────────────┘
                                                      │
        ┌─────────────────────────────────────────────┼─────────────────────────────────────────────┐
        │                                             │                                             │
        ▼                                             ▼                                             ▼
┌───────────────┐                           ┌─────────────────┐                           ┌─────────────────┐
│  IRC CLIENT   │                           │  MESSAGE ROUTER │                           │  MEMORY SYSTEM  │
│ (irc/client)  │                           │  (core/router)  │                           │    (memory/)    │
│               │                           │                 │                           │                 │
│ - Two pydle   │                           │ - 30-msg buffer │                           │ - ChromaDB      │
│   connections │                           │ - Per-channel   │                           │ - Local GPU     │
│ - Merry nick  │                           │ - Timestamps    │                           │   embeddings    │
│ - Hachi nick  │                           │                 │                           │ - Per-user      │
└───────────────┘                           └─────────────────┘                           └─────────────────┘
                                                      │
                                                      ▼
                                    ┌─────────────────────────────────────────┐
                                    │         RESPONSE PIPELINE               │
                                    └─────────────────────────────────────────┘
                                                      │
        ┌─────────────────┬───────────────────────────┼───────────────────────────┬─────────────────┐
        │                 │                           │                           │                 │
        ▼                 ▼                           ▼                           ▼                 ▼
┌───────────────┐ ┌───────────────┐         ┌─────────────────┐         ┌───────────────┐ ┌───────────────┐
│ RESPONSE GATE │ │  ENGAGEMENT   │         │  BOT SELECTOR   │         │    CONTEXT    │ │   RESPONDER   │
│ (core/gate)   │ │    CHECK      │         │ (memory/select) │         │    BUILDER    │ │ (core/respond)│
│               │ │ (core/engage) │         │                 │         │ (core/context)│ │               │
│ STATISTICAL   │ │               │         │ - Personality   │         │               │ │ - Opus 4.5    │
│ - P_base=0.05 │ │ Gemini Flash  │         │   embeddings    │         │ Gemini Flash  │ │ - Gemini Pro  │
│ - P_mention   │ │ Yes/No only   │         │ - Topic match   │         │ + RAG tools   │ │ No tools      │
│ - P_question  │ │               │         │ - Recency bias  │         │ Max 3 rounds  │ │               │
│ - P_decay     │ │               │         │                 │         │               │ │               │
└───────────────┘ └───────────────┘         └─────────────────┘         └───────────────┘ └───────────────┘
```

## Message Flow

1. **IRC Message Arrives** → `IRCClient.on_message()`
2. **Add to Buffer** → `MessageRouter.add_message()` (30-msg rolling buffer per channel)
3. **Statistical Gate** → `ResponseGate.should_respond()`
   - Calculates: `P = min(base + mention + question + conv_start, 1.0) * decay^consecutive`
   - If `random() >= P`, stop here (no response)
4. **Engagement Check** → `EngagementChecker.check()` (Gemini Flash)
   - LLM decides if this is genuine human engagement
   - Mentions override this check (always proceed if mentioned)
5. **Bot Selection** → `BotSelector.select()`
   - Embeds message, queries personality collection
   - Whoever's personality is more relevant responds
   - Recency penalty: whoever spoke last gets 0.85x score
6. **Context Building** → `ContextBuilder.build()` (Gemini Flash + tools)
   - Tools: `rag_search(query)`, `ready_to_respond(summary)`
   - Max 3 rounds of tool use
   - Gathers relevant memories and context
7. **Response Generation** → `ResponseGenerator.generate()`
   - Routes to Opus (Hachiman) or Gemini Pro (Merry)
   - No tools, just generates response
8. **Send Response** → `IRCClient.send()`
9. **Memory Extraction** → `MemoryExtractor.extract_and_store()` (background task)
   - Gemini Flash extracts memorable facts
   - Stores with intensity scores

## Tricky Parts / Gotchas

### 1. Dual IRC Connections
The `IRCClient` manages TWO separate pydle clients connecting to the same server/channels. Both receive all messages, but only one responds per message. The `is_bot_message()` check prevents infinite loops.

### 2. ChromaDB Embedding Function Interface
ChromaDB's `EmbeddingFunction` interface changed. It now expects:
- `name()` method returning a string
- `__call__(input: list[str])` for batch embedding
- `embed_query()` for single queries (we added this)
- `embed_documents()` for document batches (we added this)

The `LocalEmbeddingFunction` in `memory/embeddings.py` implements all of these.

### 3. Nomic Embed Prefixes
The `nomic-ai/nomic-embed-text-v1.5` model expects prefixes:
- Documents: `"search_document: {text}"`
- Queries: `"search_query: {text}"`

This is handled in `LocalEmbeddingFunction.embed_query()` and `embed_documents()`.

### 4. Statistical Gate Formula
The gate uses ADDITIVE factors (capped at 1.0) then MULTIPLICATIVE decay:
```python
P = min(base + mention + question + conv_start, 1.0) * (decay ** consecutive_bot_messages)
```

The decay ensures bickering naturally peters out unless humans engage.

### 5. API Key Names
The code supports both `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables. Check `main.py:load_api_keys_from_env()`.

### 6. Gemini Tool Use Format
Gemini's tool calling uses a different structure than Anthropic. See `core/context.py` for the `types.Tool` and `types.FunctionDeclaration` setup. Tool results are sent back as `types.Part.from_function_response()`.

### 7. Async Chat Sessions
The context builder uses `client.aio.chats.create()` for multi-turn tool use. This maintains conversation state across tool calls.

### 8. pydle and asyncio
**Do NOT use `pydle.ClientPool`** - its `handle_forever()` calls `asyncio.run()` internally, which crashes when there's already a running event loop. Instead:
- Connect each client directly with `await client.connect()`
- Add a delay between connections to avoid IRC rate limiting
- Let pydle's internal background task (started automatically in `connect()`) handle the read loop
- In `run_forever()`, just poll `client.connected` - don't call `handle_forever()` again

### 9. YAML Config Sections
YAML parses a section with only comments as `None`, not `{}`:
```yaml
llm:
  # comments only, no actual values
```
This causes pydantic validation errors. The `load_config()` function filters out `None` values before passing to Config.

### 10. Gemini Model Names
The preview models require the `-preview` suffix:
- `gemini-3-flash-preview` (not `gemini-3-flash`)
- `gemini-3-pro-preview` (not `gemini-3-pro`)

### 11. Nomic Embed Dependencies
The `nomic-ai/nomic-embed-text-v1.5` model requires the `einops` package, which isn't automatically pulled in by sentence-transformers.

## Configuration

Config is loaded from (in priority order):
1. Environment variables (`BICKER_*` prefix)
2. `config.yaml` file
3. Default values in pydantic models

Key settings in `config.yaml`:
```yaml
gate:
  base_prob: 0.05      # Baseline response probability
  mention_prob: 0.8    # Bonus when bot is mentioned
  question_prob: 0.3   # Bonus for questions
  decay_factor: 0.5    # Multiplied per consecutive bot message

memory:
  high_intensity_threshold: 0.7  # Memories above this are always included
```

## Testing

- **Unit tests**: Config validation, gate logic, memory operations
- **Property tests**: Hypothesis tests for probability bounds, monotonicity
- **Integration tests**: Real API calls (skipped if no keys in `.env`)

Run with:
```bash
uv run pytest tests/ -v
```

## Bot Personalities

### Merry (Gemini 3 Pro Preview)
- Direct, action-oriented dream demon
- Frustrated by overthinking
- "Just DO it already!"
- Gets impatient with Hachiman's analysis paralysis

### Hachiman (Claude Opus 4.5)
- Cynical observer, finds flaws in everything
- Self-deprecating humor
- "In other words..."
- Secretly caring under the pessimism

The personalities are defined in `personalities/merry.py` and `personalities/hachiman.py`. The bot selector uses embeddings of these personalities + recent messages to determine who responds.

## File Quick Reference

| File | Purpose |
|------|---------|
| `orchestrator.py` | Ties everything together, main processing loop |
| `irc/client.py` | Dual pydle connection management |
| `core/gate.py` | Statistical response decision |
| `core/engagement.py` | Gemini Flash engagement check |
| `core/context.py` | RAG context building with tools |
| `core/responder.py` | Final response generation |
| `memory/store.py` | ChromaDB memory storage |
| `memory/selector.py` | Topic-based bot selection |
| `memory/embeddings.py` | Local GPU embedding function |
| `memory/extractor.py` | Post-response memory extraction |
