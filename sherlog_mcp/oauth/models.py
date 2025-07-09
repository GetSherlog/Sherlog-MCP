from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class OAuthToken(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_at: datetime
    scopes: List[str] = Field(default_factory=list)
    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TokenResponse(BaseModel):
    user_id: str
    access_token: str
    expires_at: datetime
    scopes: List[str]


class AuthorizeResponse(BaseModel):
    authorization_url: str
    state: str