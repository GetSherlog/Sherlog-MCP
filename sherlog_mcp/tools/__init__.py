"""Sherlog MCP Tools Package

This package contains all the available tools for the Sherlog MCP server.
Tools are automatically registered with the FastMCP app when imported.
"""

from . import (
    cli_tools,
    code_retrieval,
)

__all__ = [
    "cli_tools",
    "code_retrieval",
]

try:
    from .cli_tools import (
        call_cli,
        search_pypi,
        query_apt_package_status,
    )
except ImportError:
    pass

try:
    from .code_retrieval import (
        configure_supported_languages,
        find_class_implementation,
        find_method_implementation,
        get_codebase_stats,
        list_all_classes,
        list_all_methods,
    )
except ImportError:
    pass
