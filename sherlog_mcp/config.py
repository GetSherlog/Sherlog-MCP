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
        description="Root logging level for the MCP server.",
        alias="LOG_LEVEL",
    )

    max_sessions: int = Field(
        default=4,
        description="Maximum number of concurrent sessions to maintain.",
        alias="MCP_MAX_SESSIONS",
    )

    auto_reset_threshold: int = Field(
        default=200,
        description="Number of operations before automatic memory cleanup.",
        alias="MCP_AUTO_RESET_THRESHOLD",
    )

    auto_reset_enabled: bool = Field(
        default=True,
        description="Enable automatic memory management for sessions.",
        alias="MCP_AUTO_RESET_ENABLED",
    )

    max_output_size: int = Field(
        default=50000,
        description="Maximum output size per buffer in bytes.",
        alias="MCP_MAX_OUTPUT_SIZE",
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

    external_mcps_json: str | None = Field(
        default=None,
        description="JSON string in Claude Desktop format containing external MCP server configurations",
        alias="EXTERNAL_MCPS_JSON",
    )

    external_mcps: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration for external MCP servers loaded from EXTERNAL_MCPS_JSON",
    )

    google_client_id: str = Field(
        default="",
        description="Google OAuth2 Client ID",
        alias="GOOGLE_CLIENT_ID",
    )

    google_client_secret: str = Field(
        default="",
        description="Google OAuth2 Client Secret",
        alias="GOOGLE_CLIENT_SECRET",
    )

    google_redirect_uri: str = Field(
        default="http://localhost:8000/auth/google/callback",
        description="Google OAuth2 Redirect URI",
        alias="GOOGLE_REDIRECT_URI",
    )

    oauth_encryption_key: str = Field(
        default="",
        description="Fernet encryption key for OAuth token storage (base64 encoded)",
        alias="OAUTH_ENCRYPTION_KEY",
    )

    oauth_storage_path: str = Field(
        default="/app/data/oauth_tokens",
        description="Path to store encrypted OAuth tokens",
        alias="OAUTH_STORAGE_PATH",
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
        except Exception:
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


settings = get_settings()