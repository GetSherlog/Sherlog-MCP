"""LogAI MCP Tools Package

This package contains all the available tools for the LogAI MCP server.
Tools are automatically registered with the FastMCP app when imported.
"""

# Import all tool modules to register them with the app
from . import data_loading
from . import preprocessing  
from . import vectorization
from . import feature_extraction
from . import clustering
from . import anomaly
from . import docker_tools
from . import filesystem_tools

# MCP tools are imported directly in server.py to avoid circular imports

__all__ = [
    "data_loading",
    "preprocessing", 
    "vectorization",
    "feature_extraction",
    "clustering",
    "anomaly",
    "docker_tools",
    "filesystem_tools",
]

from .docker_tools import list_containers, get_container_logs
from .filesystem_tools import (
    read_file,
    list_directory,
    directory_tree,
    search_files,
    get_file_info,
    list_allowed_directories
)