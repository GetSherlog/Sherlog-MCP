"""Central session & utility helpers for LogAI FastMCP server.

This module owns the FastMCP *app* instance plus the in-memory scratch-pad
(`session_vars`) that lets individual tool calls communicate with each other.
All other modules should *only* import what they need from here instead of
instantiating additional `FastMCP` objects.
"""

from typing import Any, Dict, List
import logging

import nltk
import nltk.downloader
from mcp.server.fastmcp import FastMCP
from logai_mcp.config import get_settings

app = FastMCP(name="LogAIMCP")

for _resource in [
    "tokenizers/punkt",
    "corpora/wordnet",
    "taggers/averaged_perceptron_tagger",
]:
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_resource.split("/")[-1], quiet=True)

session_vars: Dict[str, Any] = {}
session_meta: Dict[str, Dict[str, Any]] = {}

logger = logging.getLogger("LogAIMCP")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_handler)

settings = get_settings()
logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))