"""FastAPI debug server for browsing traces and debugging bot behavior."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from bicker_bot.debug.config_loader import ConfigLoader
from bicker_bot.tracing import TraceStore

if TYPE_CHECKING:
    from bicker_bot.memory.store import MemoryStore
    from bicker_bot.orchestrator import ReplayResult

# Type alias for the replay function signature
ReplayFn = Callable[[str, dict], Awaitable["ReplayResult"]]

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    trace_store: TraceStore,
    config_dir: Path | None = None,
    memory_store: MemoryStore | None = None,
    replay_fn: ReplayFn | None = None,
) -> FastAPI:
    """Create the debug server FastAPI app.

    Args:
        trace_store: The trace store for retrieving saved traces.
        config_dir: Optional directory containing prompt/policy configs.
        memory_store: Optional memory store for browsing memories (future use).
        replay_fn: Optional async function to replay a trace with config overrides.
                   Signature: async (trace_id: str, config_overrides: dict) -> ReplayResult

    Returns:
        A configured FastAPI application.
    """
    app = FastAPI(title="Bicker-Bot Debug")

    # Ensure template directory exists
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

    # Mount static files if directory exists
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Initialize config loader if config directory provided
    config_loader = None
    if config_dir and config_dir.exists():
        config_loader = ConfigLoader(config_dir)

    @app.get("/", response_class=RedirectResponse)
    async def root():
        """Redirect root to traces list."""
        return RedirectResponse(url="/traces", status_code=307)

    @app.get("/traces", response_class=HTMLResponse)
    async def traces_list(
        request: Request,
        channel: str | None = None,
        bot: str | None = None,
    ):
        """List recent traces with optional filtering."""
        traces = trace_store.recent(limit=50, channel=channel, bot=bot)
        return templates.TemplateResponse(
            request,
            "traces.html",
            {
                "traces": traces,
                "channel_filter": channel,
                "bot_filter": bot,
            },
        )

    @app.get("/traces/{trace_id}", response_class=HTMLResponse)
    async def trace_detail(request: Request, trace_id: str):
        """Show detailed view of a single trace."""
        trace = trace_store.get(trace_id)
        if trace is None:
            return HTMLResponse("Trace not found", status_code=404)
        return templates.TemplateResponse(
            request,
            "trace_detail.html",
            {"trace": trace},
        )

    @app.get("/memories", response_class=HTMLResponse)
    async def memories_list(request: Request):
        """Memory browser."""
        if memory_store is None:
            return HTMLResponse("Memory store not configured", status_code=503)
        return templates.TemplateResponse(
            request, "memories.html", {"collections": ["memories"]}
        )

    @app.get("/memories/search", response_class=HTMLResponse)
    async def memories_search(
        request: Request,
        query: str = "",
        limit: int = 20,
    ):
        """Search memories (htmx partial)."""
        if memory_store is None:
            return HTMLResponse("Memory store not configured")

        if query:
            results = memory_store.search(query=query, limit=limit)
        else:
            results = []

        return templates.TemplateResponse(
            request, "partials/memory_results.html", {"results": results, "query": query}
        )

    @app.delete("/memories/{memory_id}")
    async def delete_memory(memory_id: str):
        """Delete a memory."""
        if memory_store is None:
            return {"error": "Memory store not configured"}
        success = memory_store.delete(memory_id)
        return {"success": success}

    @app.get("/config/modal", response_class=HTMLResponse)
    async def config_modal(request: Request):
        """Config editor modal (htmx)."""
        prompts = config_loader.list_prompts() if config_loader else []
        prompt_configs = {}
        if config_loader:
            for name in prompts:
                prompt_config = config_loader.get_prompt(name)
                if prompt_config:
                    prompt_configs[name] = prompt_config
        policies = config_loader.get_policies() if config_loader else {}
        return templates.TemplateResponse(
            request,
            "partials/config_modal.html",
            {
                "prompts": prompts,
                "prompt_configs": prompt_configs,
                "policies": policies,
            },
        )

    @app.post("/config/reload")
    async def reload_config():
        """Hot-reload config from disk."""
        if config_loader:
            config_loader.reload()
            return {"success": True}
        return {"success": False, "error": "No config loader"}

    @app.post("/traces/{trace_id}/replay", response_class=HTMLResponse)
    async def replay_trace(request: Request, trace_id: str):
        """Replay a trace with current/modified config for A/B comparison."""
        if replay_fn is None:
            return HTMLResponse("Replay not available", status_code=503)

        original = trace_store.get(trace_id)
        if original is None:
            return HTMLResponse("Trace not found", status_code=404)

        try:
            # Run replay with empty config overrides (use current config)
            replay_result = await replay_fn(trace_id, {})

            return templates.TemplateResponse(
                request,
                "partials/replay_comparison.html",
                {
                    "original": replay_result.original,
                    "replayed": replay_result.replayed,
                    "decision_diffs": replay_result.decision_diffs,
                },
            )
        except Exception as e:
            logger.exception(f"Replay failed for trace {trace_id}")
            return HTMLResponse(f"Replay failed: {e}", status_code=500)

    return app
