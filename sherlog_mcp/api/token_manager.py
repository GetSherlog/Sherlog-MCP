"""
OAuth Token Manager for secure encrypted storage of OAuth tokens.
"""

import json
import os
import sqlite3
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from fastapi import HTTPException
from cryptography.fernet import Fernet


class TokenManager:
    """
    Manages OAuth tokens with encryption and SQLite database storage.
    """
    
    def __init__(self, db_path: Path | None = None):
        """
        Initialize the TokenManager.
        
        Args:
            db_path: Custom database path, defaults to /app/data/oauth/tokens.db
        """
        if db_path is None:
            self.db_path = Path("/app/data/oauth/tokens.db")
        else:
            self.db_path = db_path
        self.oauth_dir = self.db_path.parent
        self.encryption_key = self._get_or_create_key()
        self.fernet = Fernet(self.encryption_key)
        self._init_database()
    
    def _get_or_create_key(self) -> bytes:
        """
        Get encryption key from environment or create a new one.
        
        Returns:
            Encryption key as bytes
        """
        key = os.getenv("OAUTH_ENCRYPTION_KEY")
        if not key:
            key = Fernet.generate_key()
            print(f"⚠️  Generated new encryption key. Set OAUTH_ENCRYPTION_KEY={key.decode()}")
            print("   Store this key securely - you'll need it to decrypt existing tokens!")
        else:
            key = key.encode()
        return key
    
    def _init_database(self):
        """Initialize the SQLite database with the tokens table."""
        self.oauth_dir.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS oauth_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service TEXT NOT NULL UNIQUE,
                    encrypted_token_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def save_token(self, service: str, token_data: Dict) -> None:
        """
        Encrypt and save token data for a service.
        
        Args:
            service: The OAuth service name
            token_data: Dictionary containing token information
        """
        token_json = json.dumps(token_data)
        
        encrypted_data = self.fernet.encrypt(token_json.encode())
        encrypted_b64 = base64.b64encode(encrypted_data).decode()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO oauth_tokens 
                (service, encrypted_token_data, updated_at) 
                VALUES (?, ?, ?)
            ''', [service, encrypted_b64, datetime.now()])
            conn.commit()
    
    def get_token(self, service: str) -> Dict:
        """
        Retrieve and decrypt token data for a service.
        
        Args:
            service: The OAuth service name
            
        Returns:
            Dictionary containing token data
            
        Raises:
            HTTPException: If token not found or decryption fails
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT encrypted_token_data FROM oauth_tokens WHERE service = ?',
                [service]
            )
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No token found for service: {service}"
                )
            
            try:
                # Decrypt the token data
                encrypted_data = base64.b64decode(row[0])
                decrypted_json = self.fernet.decrypt(encrypted_data).decode()
                return json.loads(decrypted_json)
            except Exception as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to decrypt token for {service}: {str(e)}"
                ) from e
    
    def list_services(self) -> List[Dict]:
        """
        List all services with stored tokens.
        
        Returns:
            List of dictionaries with service and timestamp information
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT service, created_at FROM oauth_tokens')
            return [
                {"service": row[0], "created_at": row[1]} 
                for row in cursor.fetchall()
            ]
    
    def delete_token(self, service: str) -> None:
        """
        Delete token for a service.
        
        Args:
            service: The OAuth service name
            
        Raises:
            HTTPException: If token not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'DELETE FROM oauth_tokens WHERE service = ?', 
                [service]
            )
            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No token found for service: {service}"
                )
            conn.commit()
    
    def token_exists(self, service: str) -> bool:
        """
        Check if a token exists for a service.
        
        Args:
            service: The OAuth service name
            
        Returns:
            True if token exists, False otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT 1 FROM oauth_tokens WHERE service = ? LIMIT 1',
                [service]
            )
            return cursor.fetchone() is not None
    
    def get_token_metadata(self, service: str) -> Dict:
        """
        Get metadata about a token without decrypting the actual token data.
        
        Args:
            service: The OAuth service name
            
        Returns:
            Dictionary with service, created_at, and updated_at
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'SELECT service, created_at, updated_at FROM oauth_tokens WHERE service = ?',
                [service]
            )
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(
                    status_code=404,
                    detail=f"No token found for service: {service}"
                )
            
            return {
                "service": row[0],
                "created_at": row[1],
                "updated_at": row[2]
            } 