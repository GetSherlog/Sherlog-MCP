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
    code_retrieval,
)  # noqa: F401

import logging

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the MCP server"""
    logger.info("Starting LogAI MCP Server...")
    
    # Auto-restore previous session
    from logai_mcp.session import restore_session
    restore_session()
    
    # FastMCP runs on stdio transport by default
    app.run()


if __name__ == "__main__":
    main()
