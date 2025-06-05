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
from . import github_tools
from . import s3_tools
from . import cloudwatch_tools
from . import code_retrieval
from . import sentry_tools
from . import mixpanel_tools

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
    "github_tools",
    "s3_tools",
    "cloudwatch_tools",
    "code_retrieval",
    "sentry_tools",
    "mixpanel_tools",
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
from .github_tools import (
    get_issue,
    search_issues,
    get_pull_request,
    list_pull_requests,
    get_pull_request_files,
    get_pull_request_comments,
    get_pull_request_reviews,
    list_commits,
)
from .cloudwatch_tools import (
    list_log_groups,
    list_log_streams,
    query_logs,
    get_log_events,
    list_metrics,
    get_metric_statistics,
    list_alarms,
    get_alarm_history,
    put_metric_data,
)
from .code_retrieval import (
    find_method_implementation,
    find_class_implementation,
    list_all_methods,
    list_all_classes,
    get_codebase_stats,
    configure_supported_languages,
)
from .sentry_tools import (
    list_projects,
    get_sentry_issue,
    list_project_issues,
    get_sentry_event,
    list_issue_events,
    resolve_short_id,
    create_project,
    list_organization_replays,
)
from .mixpanel_tools import (
    export_events,
    query_insights,
    query_funnels,
    query_retention,
    query_people,
    list_event_names,
    list_event_properties,
)