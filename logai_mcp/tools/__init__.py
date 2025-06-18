"""LogAI MCP Tools Package

This package contains all the available tools for the LogAI MCP server.
Tools are automatically registered with the FastMCP app when imported.
"""

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

try:
    from .docker_tools import list_containers, get_container_logs
except ImportError:
    pass

from .filesystem_tools import (
    read_file,
    list_directory,
    directory_tree,
    search_files,
    get_file_info,
    list_allowed_directories
)

try:
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
except ImportError:
    pass

try:
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
except ImportError:
    pass

try:
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
except ImportError:
    pass

try:
    from .code_retrieval import (
        find_method_implementation,
        find_class_implementation,
        list_all_methods,
        list_all_classes,
        get_codebase_stats,
        configure_supported_languages,
    )
except ImportError:
    pass

try:
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
except ImportError:
    pass

try:
    from .mixpanel_tools import (
        export_events,
        query_insights,
        query_funnels,
        query_retention,
        query_people,
        list_event_names,
        list_event_properties,
    )
except ImportError:
    pass