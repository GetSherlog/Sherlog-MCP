"""
Utility functions for accessing OAuth tokens from the encrypted database.
"""

from typing import Optional, Dict, Any
from .token_manager import TokenManager


def get_oauth_token(service: str) -> Optional[Dict[str, Any]]:
    """
    Get OAuth token for a specific service.
    
    Args:
        service: The OAuth service name (e.g., 'google', 'notion', 'linear')
        
    Returns:
        Dict containing token data or None if not found
    """
    try:
        token_manager = TokenManager()
        return token_manager.get_token(service)
    except Exception:
        return None


def get_access_token(service: str) -> Optional[str]:
    """
    Get just the access token string for a specific service.
    
    Args:
        service: The OAuth service name
        
    Returns:
        Access token string or None if not found
    """
    token_data = get_oauth_token(service)
    if token_data:
        return token_data.get('access_token')
    return None


def is_token_available(service: str) -> bool:
    """
    Check if a token is available for a specific service.
    
    Args:
        service: The OAuth service name
        
    Returns:
        True if token exists, False otherwise
    """
    return get_oauth_token(service) is not None


def get_available_services() -> list:
    """
    Get list of all services with stored tokens.
    
    Returns:
        List of service names that have stored tokens
    """
    try:
        token_manager = TokenManager()
        services = token_manager.list_services()
        return [s['service'] for s in services]
    except Exception:
        return []