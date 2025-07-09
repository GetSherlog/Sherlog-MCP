import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow

from sherlog_mcp.config import settings
from sherlog_mcp.oauth.models import AuthorizeResponse, OAuthToken


class GoogleOAuthFlow:
    GOOGLE_WORKSPACE_SCOPES = [
        "https://www.googleapis.com/auth/userinfo.email",
        "https://www.googleapis.com/auth/userinfo.profile",
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/admin.directory.user.readonly",
    ]

    def __init__(self):
        self.client_config = {
            "web": {
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris": [settings.google_redirect_uri],
            }
        }
        self.state_storage: Dict[str, Dict] = {}

    def create_authorization_url(
        self, user_id: str, scopes: Optional[List[str]] = None
    ) -> AuthorizeResponse:
        if scopes is None:
            scopes = self.GOOGLE_WORKSPACE_SCOPES

        flow = Flow.from_client_config(
            self.client_config, scopes=scopes, redirect_uri=settings.google_redirect_uri
        )

        state = secrets.token_urlsafe(32)
        self.state_storage[state] = {"user_id": user_id, "scopes": scopes}

        authorization_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
            state=state,
        )

        return AuthorizeResponse(authorization_url=authorization_url, state=state)

    def handle_callback(
        self, code: str, state: str
    ) -> Optional[OAuthToken]:
        state_data = self.state_storage.pop(state, None)
        if not state_data:
            raise ValueError("Invalid state parameter")

        user_id = state_data["user_id"]
        scopes = state_data["scopes"]

        flow = Flow.from_client_config(
            self.client_config, scopes=scopes, redirect_uri=settings.google_redirect_uri
        )

        flow.fetch_token(code=code)
        credentials = flow.credentials

        expires_at = credentials.expiry
        if not expires_at:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        elif expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        return OAuthToken(
            access_token=credentials.token or "",  # type: ignore[arg-type]
            refresh_token=credentials.refresh_token or "",  # type: ignore[arg-type]
            token_type="Bearer",
            expires_at=expires_at or datetime.now(timezone.utc) + timedelta(hours=1),
            scopes=list(credentials.scopes or scopes),
            user_id=user_id,
        )

    def refresh_token(self, oauth_token: OAuthToken) -> OAuthToken:
        from google.oauth2.credentials import Credentials

        credentials = Credentials(
            token=oauth_token.access_token,
            refresh_token=oauth_token.refresh_token,
            token_uri=self.client_config["web"]["token_uri"],
            client_id=self.client_config["web"]["client_id"],
            client_secret=self.client_config["web"]["client_secret"],
            scopes=oauth_token.scopes,
        )

        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())

            expires_at = credentials.expiry
            if not expires_at:
                expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
            elif expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)

            oauth_token.access_token = credentials.token
            oauth_token.expires_at = expires_at or datetime.now(timezone.utc) + timedelta(
                hours=1
            )
            oauth_token.updated_at = datetime.now(timezone.utc)

        return oauth_token

