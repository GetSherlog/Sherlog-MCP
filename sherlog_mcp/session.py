"""Central session & utility helpers for Sherlog FastMCP server.

This module owns the FastMCP *app* instance plus the in-memory scratch-pad
(`session_vars`) that lets individual tool calls communicate with each other.
All other modules should *only* import what they need from here instead of
instantiating additional `FastMCP` objects.
"""

import json
import logging
from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import polars as pl

import pydantic_core
from fastmcp import FastMCP

from sherlog_mcp.config import get_settings
from sherlog_mcp.middleware import SessionMiddleware

_original_to_json = pydantic_core.to_json


def _enhanced_to_json(value, *, fallback=None, **kwargs):
    """Enhanced JSON serializer that handles pandas/numpy objects."""

    def _convert_scientific_objects(obj):
        """Convert scientific objects to JSON-serializable formats."""
        if isinstance(obj, pd.DataFrame):
            df_clean = obj.replace([np.inf, -np.inf], ['Infinity', '-Infinity'])
            df_clean = df_clean.where(pd.notnull(df_clean), None)
            try:
                return json.loads(df_clean.to_json(orient="records", date_format="iso"))
            except (ValueError, TypeError):
                return df_clean.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            series_clean = obj.replace([np.inf, -np.inf], ['Infinity', '-Infinity'])
            series_clean = series_clean.where(pd.notnull(series_clean), None)
            try:
                return json.loads(series_clean.to_json(date_format="iso"))
            except (ValueError, TypeError):
                return series_clean.to_dict()
        elif isinstance(obj, np.ndarray):
            if obj.dtype.kind in "fc":
                numeric = obj.copy()
                nan_mask = np.isnan(numeric)
                pos_inf_mask = np.isinf(numeric) & (numeric > 0)
                neg_inf_mask = np.isinf(numeric) & (numeric < 0)

                converted = numeric.astype(object)
                converted[nan_mask] = None
                converted[pos_inf_mask] = "Infinity"
                converted[neg_inf_mask] = "-Infinity"

                return converted.tolist()
            else:
                return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return 'Infinity' if obj > 0 else '-Infinity'
            return obj.item()
        elif isinstance(obj, pl.DataFrame):
            return obj.to_dicts()

        return obj

    try:
        converted_value = _convert_scientific_objects(value)
        if converted_value is not value:
            return _original_to_json(converted_value, fallback=fallback, **kwargs)
    except Exception as e:
        if fallback is not None:
            try:
                return _original_to_json(fallback(value), **kwargs)
            except Exception:
                pass

    return _original_to_json(value, fallback=fallback, **kwargs)


pydantic_core.to_json = _enhanced_to_json


@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for graceful startup and shutdown."""
    # Startup
    logger.info("Starting Sherlog MCP server with optimized session persistence")
    
    # Initialize session middleware - get the instance from the app's middlewares
    try:
        from sherlog_mcp.middleware.session_middleware import start_persistence_worker
        await start_persistence_worker()
    except Exception as e:
        logger.error(f"Error during session middleware startup: {e}")
    
    yield
    # Shutdown
    logger.info("Shutting down Sherlog MCP server...")
    try:
        from sherlog_mcp.middleware.session_middleware import shutdown_persistence
        await shutdown_persistence()
        logger.info("Session persistence shutdown complete")
    except Exception as e:
        logger.error(f"Error during session persistence shutdown: {e}")


app = FastMCP(name="SherlogMCP", lifespan=lifespan)

settings = get_settings()
session_middleware = SessionMiddleware(max_sessions=settings.max_sessions)
app.add_middleware(session_middleware)


@app.custom_route("/health", methods=["GET"])
async def health_check(request):
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "ok", "service": "Sherlog MCP"})

session_vars: dict[str, Any] = {}
session_meta: dict[str, dict[str, Any]] = {}

logger = logging.getLogger("SherlogMCP")
if not logger.handlers:
    import sys
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(_handler)

settings = get_settings()
logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))


SESSIONS_DIR = Path("/app/data/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
