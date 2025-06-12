from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional, List, Dict, Any
import json
import os
from pathlib import Path


class Settings(BaseSettings): # type: ignore[misc]
    """Central application configuration loaded from environment variables.

    Add new configuration options here as regular attributes.  Values can be
    supplied via the environment (see the *env* parameter on each Field) or a
    local .env file in the project root.
    """

    log_level: str = Field(
        default="INFO",
        description="Root logging level for the LogAI-MCP server.",
        alias="LOGAI_MCP_LOG_LEVEL",
    )

    data_directory: str = Field(
        default="data",
        description="Default directory where data files are stored/read.",
        alias="LOGAI_MCP_DATA_DIR",
    )

    confluence_url: Optional[str] = Field(
        default=None,
        description="Base URL for the Confluence instance (e.g., https://your-company.atlassian.net/wiki)",
        alias="CONFLUENCE_URL",
    )

    confluence_username: Optional[str] = Field(
        default=None,
        description="Username/email used for Confluence API authentication.",
        alias="CONFLUENCE_USERNAME",
    )

    confluence_api_token: Optional[str] = Field(
        default=None,
        description="API token for Confluence.",
        alias="CONFLUENCE_API_TOKEN",
    )

    jira_url: Optional[str] = Field(
        default=None,
        description="Base URL for the Jira instance (e.g., https://your-company.atlassian.net)",
        alias="JIRA_URL",
    )

    jira_username: Optional[str] = Field(
        default=None,
        description="Username/email used for Jira API authentication.",
        alias="JIRA_USERNAME",
    )

    jira_api_token: Optional[str] = Field(
        default=None,
        description="API token for Jira.",
        alias="JIRA_API_TOKEN",
    )

    github_pat_token: Optional[str] = Field(
        default=None,
        description="Personal Access Token for GitHub API authentication.",
        alias="GITHUB_PAT_TOKEN",
    )

    grafana_url: Optional[str] = Field(
        default=None,
        description="Base URL for the Grafana instance (e.g., http://localhost:3000)",
        alias="GRAFANA_URL",
    )

    grafana_api_key: Optional[str] = Field(
        default=None,
        description="API key for Grafana authentication.",
        alias="GRAFANA_API_KEY",
    )

    # Sentry Configuration
    sentry_auth_token: Optional[str] = Field(
        default=None,
        description="Authentication token for Sentry API access.",
        alias="SENTRY_AUTH_TOKEN",
    )

    sentry_host: Optional[str] = Field(
        default="https://sentry.io",
        description="Base URL for the Sentry instance (e.g., https://sentry.io or https://your-domain.sentry.io)",
        alias="SENTRY_HOST",
    )

    # MindsDB Configuration
    mindsdb_url: Optional[str] = Field(
        default=None,
        description="Base URL for the MindsDB instance (e.g., http://localhost:47334)",
        alias="MINDSDB_URL",
    )

    mindsdb_access_token: Optional[str] = Field(
        default=None,
        description="Access token for MindsDB MCP server authentication.",
        alias="MINDSDB_ACCESS_TOKEN",
    )

    # AWS S3 Configuration
    aws_access_key_id: Optional[str] = Field(
        default=None,
        description="AWS Access Key ID for S3 authentication.",
        alias="AWS_ACCESS_KEY_ID",
    )

    aws_secret_access_key: Optional[str] = Field(
        default=None,
        description="AWS Secret Access Key for S3 authentication.",
        alias="AWS_SECRET_ACCESS_KEY",
    )

    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for S3 operations.",
        alias="AWS_REGION",
    )

    aws_session_token: Optional[str] = Field(
        default=None,
        description="AWS Session Token for temporary credentials (optional).",
        alias="AWS_SESSION_TOKEN",
    )

    # Mixpanel Configuration
    mixpanel_api_secret: Optional[str] = Field(
        default=None,
        description="API secret for Mixpanel authentication.",
        alias="MIXPANEL_API_SECRET",
    )

    mixpanel_host: Optional[str] = Field(
        default="https://mixpanel.com",
        description="Base URL for the Mixpanel instance (e.g., https://mixpanel.com or https://eu.mixpanel.com)",
        alias="MIXPANEL_HOST",
    )

    # Code retrieval configuration
    codebase_path: Optional[str] = Field(
        default=None,
        description="Path to the codebase directory for code retrieval tools.",
        alias="CODEBASE_PATH",
    )

    supported_languages: List[str] = Field(
        default=["java", "kotlin", "python", "typescript", "javascript", "cpp", "rust"],
        description="List of programming languages to analyze in the codebase. "
                    "Supported: java, kotlin, python, typescript, javascript, cpp, rust",
        alias="SUPPORTED_LANGUAGES",
    )

    # Kubernetes Configuration
    kubeconfig_path: Optional[str] = Field(
        default=None,
        description="Path to the Kubernetes config file. "
                    "If not provided, will use default kubeconfig or in-cluster config.",
        alias="KUBECONFIG_PATH",
    )

    # MCP Configuration File Path
    mcp_config_path: str = Field(
        default="mcp.json",
        description="Path to the MCP configuration file (similar to Claude Desktop's config)",
        alias="MCP_CONFIG_PATH",
    )
    
    # External MCP Configuration (loaded from mcp.json)
    external_mcps: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration for external MCP servers loaded from mcp.json",
    )

    def load_mcp_config(self) -> Dict[str, Dict[str, Any]]:
        """Load MCP configuration from mcp.json file."""
        config_path = Path(self.mcp_config_path)

        # Check multiple locations for mcp.json
        search_paths = [
            config_path if config_path.is_absolute() else None,
            Path.cwd() / config_path,
            Path.home() / ".logai-mcp" / "mcp.json",
            Path(__file__).parent.parent / config_path,
        ]

        for path in search_paths:
            if path and path.exists():
                try:
                    with open(path, 'r') as f:
                        config = json.load(f)
                        # Support both "mcpServers" (Claude Desktop style) and direct format
                        if "mcpServers" in config:
                            return config["mcpServers"]
                        return config
                except Exception as e:
                    print(f"Error loading MCP config from {path}: {e}")

        return {}

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """Return a cached `Settings` instance.

    Using an LRU cache guarantees that the environment is read only once and
    the same `Settings` object is reused everywhere, acting as a lightweight
    singleton without the pitfalls of global state.
    """
    settings = Settings()

    # Load external MCPs from mcp.json
    settings.external_mcps = settings.load_mcp_config()

    return settings