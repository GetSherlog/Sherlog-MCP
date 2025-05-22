from functools import lru_cache
from pydantic import BaseSettings, Field
from typing import Optional


class Settings(BaseSettings): # type: ignore[misc]
    """Central application configuration loaded from environment variables.

    Add new configuration options here as regular attributes.  Values can be
    supplied via the environment (see the *env* parameter on each Field) or a
    local .env file in the project root.
    """

    # ---------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Root logging level for the LogAI-MCP server.",
    )

    # ---------------------------------------------------------------------
    # Data paths
    # ---------------------------------------------------------------------
    data_directory: str = Field(
        default="data",
        description="Default directory where data files are stored/read.",
    )

    # ---------------------------------------------------------------------
    # Atlassian – Confluence & Jira integration
    # ---------------------------------------------------------------------
    confluence_url: Optional[str] = Field(
        default=None,
        description="Base URL for the Confluence instance (e.g., https://your-company.atlassian.net/wiki)",
    )

    confluence_username: Optional[str] = Field(
        default=None,
        description="Username/email used for Confluence API authentication.",
    )

    confluence_api_token: Optional[str] = Field(
        default=None,
        description="API token for Confluence.",
    )

    jira_url: Optional[str] = Field(
        default=None,
        description="Base URL for the Jira instance (e.g., https://your-company.atlassian.net)",
    )

    jira_username: Optional[str] = Field(
        default=None,
        description="Username/email used for Jira API authentication.",
    )

    jira_api_token: Optional[str] = Field(
        default=None,
        description="API token for Jira.",
    )

    class Config:
        # Read optional .env file in the project root (next to pyproject.toml)
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        fields = {
            "log_level": {"env": "LOGAI_MCP_LOG_LEVEL"},
            "data_directory": {"env": "LOGAI_MCP_DATA_DIR"},
            "confluence_url": {"env": "CONFLUENCE_URL"},
            "confluence_username": {"env": "CONFLUENCE_USERNAME"},
            "confluence_api_token": {"env": "CONFLUENCE_API_TOKEN"},
            "jira_url": {"env": "JIRA_URL"},
            "jira_username": {"env": "JIRA_USERNAME"},
            "jira_api_token": {"env": "JIRA_API_TOKEN"},
        }


@lru_cache()
def get_settings() -> Settings:
    """Return a cached `Settings` instance.

    Using an LRU cache guarantees that the environment is read only once and
    the same `Settings` object is reused everywhere, acting as a lightweight
    singleton without the pitfalls of global state.
    """

    return Settings() 