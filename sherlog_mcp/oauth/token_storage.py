import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet

from sherlog_mcp.config import settings
from sherlog_mcp.oauth.models import OAuthToken


class TokenStorage:
    def __init__(self):
        self.storage_path = Path(settings.oauth_storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        if settings.oauth_encryption_key:
            try:
                self._fernet = Fernet(settings.oauth_encryption_key.encode())
            except ValueError as e:
                print(f"Warning: Invalid OAUTH_ENCRYPTION_KEY format ({e}). Generating a new key.")
                self._fernet = Fernet(Fernet.generate_key())
        else:
            self._fernet = Fernet(Fernet.generate_key())

    def _get_token_path(self, user_id: str) -> Path:
        return self.storage_path / f"{user_id}_google.json"

    def _encrypt_data(self, data: str) -> bytes:
        return self._fernet.encrypt(data.encode())

    def _decrypt_data(self, encrypted_data: bytes) -> str:
        return self._fernet.decrypt(encrypted_data).decode()

    def save_token(self, oauth_token: OAuthToken) -> None:
        token_path = self._get_token_path(oauth_token.user_id)
        
        token_data = oauth_token.model_dump(mode="json")
        token_data["expires_at"] = oauth_token.expires_at.isoformat()
        token_data["created_at"] = oauth_token.created_at.isoformat()
        token_data["updated_at"] = oauth_token.updated_at.isoformat()
        
        json_data = json.dumps(token_data)
        encrypted_data = self._encrypt_data(json_data)
        
        with open(token_path, "wb") as f:
            f.write(encrypted_data)

    def load_token(self, user_id: str) -> Optional[OAuthToken]:
        token_path = self._get_token_path(user_id)
        
        if not token_path.exists():
            return None
        
        try:
            with open(token_path, "rb") as f:
                encrypted_data = f.read()
            
            json_data = self._decrypt_data(encrypted_data)
            token_data = json.loads(json_data)
            
            token_data["expires_at"] = datetime.fromisoformat(token_data["expires_at"])
            token_data["created_at"] = datetime.fromisoformat(token_data["created_at"])
            token_data["updated_at"] = datetime.fromisoformat(token_data["updated_at"])
            
            return OAuthToken(**token_data)
        except Exception:
            return None

