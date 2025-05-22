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


def main():
    """Run the FastMCP server (stdio transport)"""
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
