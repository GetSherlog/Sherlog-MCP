import json
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):  # type: ignore[misc]
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

    confluence_url: str | None = Field(
        default=None,
        description="Base URL for the Confluence instance (e.g., https://your-company.atlassian.net/wiki)",
        alias="CONFLUENCE_URL",
    )

    confluence_username: str | None = Field(
        default=None,
        description="Username/email used for Confluence API authentication.",
        alias="CONFLUENCE_USERNAME",
    )

    confluence_api_token: str | None = Field(
        default=None,
        description="API token for Confluence.",
        alias="CONFLUENCE_API_TOKEN",
    )

    jira_url: str | None = Field(
        default=None,
        description="Base URL for the Jira instance (e.g., https://your-company.atlassian.net)",
        alias="JIRA_URL",
    )

    jira_username: str | None = Field(
        default=None,
        description="Username/email used for Jira API authentication.",
        alias="JIRA_USERNAME",
    )

    jira_api_token: str | None = Field(
        default=None,
        description="API token for Jira.",
        alias="JIRA_API_TOKEN",
    )

    github_pat_token: str | None = Field(
        default=None,
        description="Personal Access Token for GitHub API authentication.",
        alias="GITHUB_PAT_TOKEN",
    )

    grafana_url: str | None = Field(
        default=None,
        description="Base URL for the Grafana instance (e.g., http://localhost:3000)",
        alias="GRAFANA_URL",
    )

    grafana_api_key: str | None = Field(
        default=None,
        description="API key for Grafana authentication.",
        alias="GRAFANA_API_KEY",
    )

    aws_access_key_id: str | None = Field(
        default=None,
        description="AWS Access Key ID for S3 authentication.",
        alias="AWS_ACCESS_KEY_ID",
    )

    aws_secret_access_key: str | None = Field(
        default=None,
        description="AWS Secret Access Key for S3 authentication.",
        alias="AWS_SECRET_ACCESS_KEY",
    )

    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for S3 operations.",
        alias="AWS_REGION",
    )

    aws_session_token: str | None = Field(
        default=None,
        description="AWS Session Token for temporary credentials (optional).",
        alias="AWS_SESSION_TOKEN",
    )

    mixpanel_api_secret: str | None = Field(
        default=None,
        description="API secret for Mixpanel authentication.",
        alias="MIXPANEL_API_SECRET",
    )

    mixpanel_host: str | None = Field(
        default="https://mixpanel.com",
        description="Base URL for the Mixpanel instance (e.g., https://mixpanel.com or https://eu.mixpanel.com)",
        alias="MIXPANEL_HOST",
    )

    codebase_path: str | None = Field(
        default=None,
        description="Path to the codebase directory for code retrieval tools.",
        alias="CODEBASE_PATH",
    )

    supported_languages: list[str] = Field(
        default=["java", "kotlin", "python", "typescript", "javascript", "cpp", "rust"],
        description="List of programming languages to analyze in the codebase. "
        "Supported: java, kotlin, python, typescript, javascript, cpp, rust",
        alias="SUPPORTED_LANGUAGES",
    )

    kubeconfig_path: str | None = Field(
        default=None,
        description="Path to the Kubernetes config file. "
        "If not provided, will use default kubeconfig or in-cluster config.",
        alias="KUBECONFIG_PATH",
    )

    external_mcps_json: str | None = Field(
        default=None,
        description="JSON string in Claude Desktop format containing external MCP server configurations",
        alias="EXTERNAL_MCPS_JSON",
    )

    external_mcps: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration for external MCP servers loaded from EXTERNAL_MCPS_JSON",
    )

    def load_mcp_config(self) -> dict[str, dict[str, Any]]:
        """Load MCP configuration from EXTERNAL_MCPS_JSON environment variable.

        Returns:
            Dictionary of external MCP server configurations

        """
        if not self.external_mcps_json:
            return {}

        try:
            config = json.loads(self.external_mcps_json)
            if isinstance(config, dict) and "mcpServers" in config:
                return config["mcpServers"]
            elif isinstance(config, dict):
                return config
            else:
                raise ValueError("EXTERNAL_MCPS_JSON must be a JSON object")
        except Exception as e:
            return {}

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache
def get_settings() -> Settings:
    """Return a cached `Settings` instance.

    Using an LRU cache guarantees that the environment is read only once and
    the same `Settings` object is reused everywhere, acting as a lightweight
    singleton without the pitfalls of global state.
    """
    settings = Settings()

    settings.external_mcps = settings.load_mcp_config()

    return settings
