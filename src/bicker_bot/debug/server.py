"""FastAPI debug server for browsing traces and debugging bot behavior."""

import logging
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from bicker_bot.debug.config_loader import ConfigLoader
from bicker_bot.tracing import TraceStore

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    trace_store: TraceStore,
    config_dir: Path | None = None,
    memory_store=None,
    replay_fn: Callable | None = None,
) -> FastAPI:
    """Create the debug server FastAPI app.

    Args:
        trace_store: The trace store for retrieving saved traces.
        config_dir: Optional directory containing prompt/policy configs.
        memory_store: Optional memory store for browsing memories (future use).
        replay_fn: Optional function to replay a trace (future use).

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

    return app
