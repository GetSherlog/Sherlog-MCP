import asyncio
import logging
import os
import uvicorn

from sherlog_mcp.session import (
    app,  # noqa: F401 – side-effect: create singleton & basic tools
)
from sherlog_mcp.tools import (
    external_mcp_tools,
)  # noqa: F401

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the MCP server"""
    logger.info("Starting Sherlog MCP Server...")

    logger.info("Registering external MCP tools...")
    try:
        asyncio.run(external_mcp_tools.auto_register_external_mcps())
        logger.info("External MCP registration complete")
    except Exception as e:
        logger.error(f"Failed to register external MCPs: {e}")

    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))
    
    logger.info(f"Starting Sherlog MCP server on {host}:{port}")
    logger.info("Transport: streamable-http (stateful)")
    http_app  = app.http_app()

    uvicorn.run(http_app, host=host, port=port)

if __name__ == "__main__":
    main()
