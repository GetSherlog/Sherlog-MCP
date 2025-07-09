import asyncio
import logging
import os
import uvicorn

import sherlog_mcp.session  # noqa: F401
from sherlog_mcp.tools import external_mcp_tools
from sherlog_mcp.main import api_app

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the MCP server"""
    logger.info("Starting Sherlog MCP Server with OAuth...")

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
    logger.info("OAuth endpoints available at /auth/google/*")
    logger.info("MCP endpoint available at /mcp")

    uvicorn.run(api_app, host=host, port=port)

if __name__ == "__main__":
    main()
