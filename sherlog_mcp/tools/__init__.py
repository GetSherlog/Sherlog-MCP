"""Sherlog MCP Tools Package

This package contains all the available tools for the Sherlog MCP server.
Tools are automatically registered with the FastMCP app when imported.
"""

from . import (
    anomaly,
    clustering,
    code_retrieval,
    data_loading,
    docker_tools,
    feature_extraction,
    filesystem_tools,
    github_tools,
    grafana_tools,
    mixpanel_tools,
    preprocessing,
    s3_tools,
    vectorization,
)

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
    "github_tools",
    "s3_tools",
    "code_retrieval",
    "mixpanel_tools",
]

try:
    from .docker_tools import get_container_logs, list_containers
except ImportError:
    pass

try:
    from .filesystem_tools import (
        directory_tree,
        get_file_info,
        list_allowed_directories,
        list_directory,
        read_file,
        search_files,
    )
except ImportError:
    pass

try:
    from .grafana_tools import (
        list_loki_label_names,
        list_loki_label_values,
        list_prometheus_label_names,
        list_prometheus_label_values,
        list_prometheus_metric_metadata,
        list_prometheus_metric_names,
        query_loki_logs,
        query_loki_stats,
        query_prometheus,
    )
except ImportError:
    pass

try:
    from .github_tools import (
        get_issue,
        get_pull_request,
        get_pull_request_comments,
        get_pull_request_files,
        get_pull_request_reviews,
        list_commits,
        list_pull_requests,
        search_issues,
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

try:
    from .mixpanel_tools import (
        export_events,
        list_event_names,
        list_event_properties,
        query_funnels,
        query_insights,
        query_people,
        query_retention,
    )
except ImportError:
    pass
