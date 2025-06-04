from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, List


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

    # Code retrieval configuration
    codebase_path: Optional[str] = Field(
        default=None,
        description="Path to the codebase directory for code retrieval tools.",
        alias="CODEBASE_PATH",
    )

    supported_languages: List[str] = Field(
        default=["java", "kotlin", "python", "typescript", "javascript", "cpp", "rust"],
        description="List of programming languages to analyze in the codebase. Supported: java, kotlin, python, typescript, javascript, cpp, rust",
        alias="SUPPORTED_LANGUAGES",
    )

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

    return Settings() 