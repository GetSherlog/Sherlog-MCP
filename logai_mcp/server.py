import asyncio
import logging
import os

from logai_mcp.session import app  # noqa: F401 – side-effect: create singleton & basic tools

# Import all tool sub-modules so their @app.tool() functions register
from logai_mcp.tools import (
    data_loading,
    preprocessing,
    vectorization,
    feature_extraction,
    clustering,
    anomaly,
    docker_tools,
    filesystem_tools,
    s3_tools,
    github_tools,
    grafana_tools,
    cloudwatch_tools,
    code_retrieval,
    sentry_tools,
    external_mcp_tools,  # Import external MCP tools module
)  # noqa: F401

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the MCP server"""
    logger.info("Starting LogAI MCP Server...")
    
    # Get transport configuration from environment variables
    transport = os.getenv('MCP_TRANSPORT', 'streamable-http')
    
    from logai_mcp.session import restore_session
    restore_session()
    
    if transport == "streamable-http":
        logger.info("Registering external MCP tools...")
        try:
            asyncio.run(external_mcp_tools.auto_register_external_mcps())
            logger.info("External MCP registration complete")
        except Exception as e:
            logger.error(f"Failed to register external MCPs: {e}")
    else:
        logger.info("Skipping external MCP registration for STDIO transport")
    
    logger.info(f"Starting LogAI MCP server with transport: {transport}")
    
    if transport == "stdio":
        # For STDIO, no additional parameters needed
        app.run(transport="stdio")
    else:
        app.run(transport="streamable-http")


if __name__ == "__main__":
    main()
