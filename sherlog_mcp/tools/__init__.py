"""Sherlog MCP Tools Package.

Importing this package triggers registration of built-in tools. Android-specific
tools are only loaded when explicitly enabled via the environment variable
`SHERLOG_ENABLE_ANDROID_TOOLS`.
"""

import os

# Feature flag for Android tools (enabled only in Android-capable images)
_ENABLE_ANDROID_TOOLS = os.getenv("SHERLOG_ENABLE_ANDROID_TOOLS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Always import core tool modules
from . import cli_tools, code_retrieval  # noqa: F401

__all__ = ["cli_tools", "code_retrieval"]

try:
    from .cli_tools import (  # noqa: F401
        call_cli,
        search_pypi,
        query_apt_package_status,
    )
except ImportError:
    pass

try:
    from .code_retrieval import (  # noqa: F401
        configure_supported_languages,
        find_class_implementation,
        find_method_implementation,
        get_codebase_stats,
        list_all_classes,
        list_all_methods,
    )
except ImportError:
    pass

# Conditionally import Android tools
if _ENABLE_ANDROID_TOOLS:
    try:
        from . import android_jobs  # noqa: F401
        __all__.append("android_jobs")
    except Exception:
        # If Android runtime is not ready, skip loading Android tools
        pass
