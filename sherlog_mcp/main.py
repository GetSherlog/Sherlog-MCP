import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
import uvicorn

from sherlog_mcp.config import settings
from sherlog_mcp.oauth import GoogleOAuthFlow, TokenResponse, TokenStorage
from sherlog_mcp.session import app as mcp_app
from sherlog_mcp.tools import external_mcp_tools
from sherlog_mcp import tools as internal_tools

logger = logging.getLogger(__name__)

google_oauth = GoogleOAuthFlow()
token_storage = TokenStorage()

mcp_app = mcp_app.http_app(path='/mcp')

api_app = FastAPI(
    title="Sherlog MCP Server with OAuth",
    description="MCP server with OAuth integrations",
    version="0.1.0",
    lifespan=mcp_app.lifespan,
)

api_app.mount("/mcp-server", mcp_app)


@api_app.get("/")
async def root():
    return {
        "message": "Sherlog MCP Server",
        "mcp_endpoint": "/mcp",
        "oauth_endpoints": {
            "authorize": "/auth/google/authorize",
            "callback": "/auth/google/callback",
            "token": "/auth/google/token/{user_id}",
        },
    }


@api_app.get("/auth/google/authorize")
async def google_authorize(
    user_id: str = Query(..., description="Unique identifier for the user"),
    scopes: Optional[str] = Query(
        None, description="Comma-separated list of Google OAuth scopes"
    ),
):
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(
            status_code=500,
            detail="Google OAuth credentials not configured",
        )

    scope_list = None
    if scopes:
        scope_list = [s.strip() for s in scopes.split(",")]

    try:
        auth_response = google_oauth.create_authorization_url(user_id, scope_list)
        return RedirectResponse(url=auth_response.authorization_url, status_code=302)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_app.get("/auth/google/callback")
async def google_callback(
    code: str = Query(..., description="Authorization code from Google"),
    state: str = Query(..., description="State parameter for CSRF protection"),
):
    try:
        oauth_token = google_oauth.handle_callback(code, state)
        if not oauth_token:
            raise HTTPException(status_code=400, detail="Failed to exchange code for token")

        if oauth_token.expires_at <= datetime.now(timezone.utc):
            oauth_token = google_oauth.refresh_token(oauth_token)

        token_storage.save_token(oauth_token)

        return {
            "message": "Successfully authenticated",
            "user_id": oauth_token.user_id,
            "scopes": oauth_token.scopes,
            "expires_at": oauth_token.expires_at.isoformat(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_app.get("/auth/google/token/{user_id}", response_model=TokenResponse)
async def get_google_token(user_id: str):
    oauth_token = token_storage.load_token(user_id)
    if not oauth_token:
        raise HTTPException(status_code=404, detail="Token not found for user")

    if oauth_token.expires_at <= datetime.now(timezone.utc) and oauth_token.refresh_token:
        try:
            oauth_token = google_oauth.refresh_token(oauth_token)
            token_storage.save_token(oauth_token)
        except Exception:
            raise HTTPException(status_code=401, detail="Token expired and refresh failed")

    return TokenResponse(
        user_id=oauth_token.user_id,
        access_token=oauth_token.access_token,
        expires_at=oauth_token.expires_at,
        scopes=oauth_token.scopes,
    )


def main():
    """Main entry point for the MCP server with OAuth"""
    logger.info("Starting Sherlog MCP Server with OAuth...")

    # Ensure built-in tools are imported so their @app.tool() registrations run
    try:
        loaded = getattr(internal_tools, "__all__", [])
        logger.info(f"Registered internal tool modules: {loaded}")
    except Exception as e:
        logger.warning(f"Failed to load internal tools: {e}")

    logger.info("Registering external MCP tools...")
    try:
        asyncio.run(external_mcp_tools.auto_register_external_mcps())
        logger.info("External MCP registration complete")
    except Exception as e:
        logger.error(f"Failed to register external MCPs: {e}")

    import os
    host = os.getenv("MCP_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info("OAuth endpoints available at /auth/google/*")
    logger.info("MCP endpoint available at /mcp")

    uvicorn.run(api_app, host=host, port=port)


if __name__ == "__main__":
    main()