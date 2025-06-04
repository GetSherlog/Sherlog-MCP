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
from . import grafana_tools
from . import s3_tools
from . import code_retrieval

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
    "grafana_tools",
    "s3_tools",
    "code_retrieval",
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
from .grafana_tools import (
    query_prometheus,
    list_prometheus_metric_metadata,
    list_prometheus_metric_names,
    list_prometheus_label_names,
    list_prometheus_label_values,
    query_loki_logs,
    list_loki_label_names,
    list_loki_label_values,
    query_loki_stats,
)
from .code_retrieval import (
    find_method_implementation,
    find_class_implementation,
    list_all_methods,
    list_all_classes,
    get_codebase_stats,
    configure_supported_languages,
)