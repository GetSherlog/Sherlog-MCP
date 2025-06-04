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
)  # noqa: F401

# Import MCP integration tools
from logai_mcp import mcp_tools  # noqa: F401
from logai_mcp.mcp_manager import initialize_mcp_manager
from logai_mcp.shell_integration import update_shell_functions
import asyncio
import logging

logger = logging.getLogger(__name__)

async def initialize_mcp_integration():
    """Initialize MCP manager and load configured clients"""
    try:
        # Initialize the MCP manager
        mcp_manager = initialize_mcp_manager(app)
        
        # Load and connect to configured MCP clients
        await mcp_manager.initialize_from_config()
        
        # Update shell with convenience functions
        await update_shell_functions()
        
        logger.info("MCP integration initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize MCP integration: {e}")

def main():
    """Run the FastMCP server (stdio transport)"""
    # Initialize MCP integration before starting server
    asyncio.create_task(initialize_mcp_integration())
    
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
