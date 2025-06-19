import asyncio
import logging
import os

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

    transport = os.getenv("MCP_TRANSPORT", "streamable-http")

    from sherlog_mcp.session import restore_session

    restore_session()

    logger.info("Registering external MCP tools...")
    try:
        asyncio.run(external_mcp_tools.auto_register_external_mcps())
        logger.info("External MCP registration complete")
    except Exception as e:
        logger.error(f"Failed to register external MCPs: {e}")

    logger.info(f"Starting Sherlog MCP server with transport: {transport}")

    if transport == "stdio":
        app.run(transport="stdio")
    else:
        app.run(transport="streamable-http")


if __name__ == "__main__":
    main()
