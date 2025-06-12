import asyncio
import logging

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
    
    # Auto-restore previous session
    from logai_mcp.session import restore_session
    restore_session()
    
    # Register external MCPs
    logger.info("Registering external MCP tools...")
    try:
        # Run the async registration in a sync context
        asyncio.run(external_mcp_tools.auto_register_external_mcps())
        logger.info("External MCP registration complete")
    except Exception as e:
        logger.error(f"Failed to register external MCPs: {e}")
    
    # Use streamable-http transport for Sherlog Canvas integration
    logger.info("Starting MCP server with streamable-http transport for web applications")
    app.run(transport="streamable-http")


if __name__ == "__main__":
    main()
