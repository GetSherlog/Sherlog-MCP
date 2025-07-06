import os
import subprocess
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import httpx
from urllib.parse import urlencode
from .oauth_config import OAUTH_CONFIG
from .token_manager import TokenManager

router = APIRouter()

token_manager = TokenManager()

class CommandRequest(BaseModel):
    command: str

@router.post("/api/execute")
async def execute_command(req: CommandRequest):
    try:
        result = subprocess.run(
            req.command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        return {
            "command": req.command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Command timed out") from None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.get("/api/oauth/services")
async def get_oauth_services():
    return {
        "services": [
            {"name": "google", "configured": bool(os.getenv("GOOGLE_CLIENT_ID"))},
            {"name": "notion", "configured": bool(os.getenv("NOTION_CLIENT_ID"))},
            {"name": "linear", "configured": bool(os.getenv("LINEAR_CLIENT_ID"))}
        ]
    }

@router.get("/api/oauth/tokens")
async def list_stored_tokens():
    """List all stored OAuth tokens"""
    return {"tokens": token_manager.list_services()}

@router.get("/api/oauth/tokens/{service}")
async def get_oauth_token(service: str):
    """Get OAuth token for a specific service"""
    try:
        token_data = token_manager.get_token(service)
        return {"service": service, "token_data": token_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.delete("/api/oauth/tokens/{service}")
async def delete_oauth_token(service: str):
    """Delete OAuth token for a specific service"""
    try:
        token_manager.delete_token(service)
        return {"message": f"Token for {service} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@router.get("/auth/{service}")
async def start_oauth(service: str):
    if service not in OAUTH_CONFIG:
        raise HTTPException(status_code=404, detail="Service not supported")

    config = OAUTH_CONFIG[service]
    redirect_uri = f"http://localhost:8000/auth/{service}/callback"

    params = {
        "client_id": config["client_id"],
        "redirect_uri": redirect_uri,
        "response_type": "code",
    }

    if "scope" in config:
        params["scope"] = config["scope"]

    auth_url = f"{config['auth_url']}?{urlencode(params)}"
    return RedirectResponse(auth_url)

@router.get("/auth/{service}/callback")
async def oauth_callback(service: str, code: str):
    if service not in OAUTH_CONFIG:
        raise HTTPException(status_code=404, detail="Service not supported")

    config = OAUTH_CONFIG[service]
    redirect_uri = f"http://localhost:8000/auth/{service}/callback"

    async with httpx.AsyncClient() as client:
        response = await client.post(
            config["token_url"],
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": config["client_id"],
                "client_secret": config["client_secret"]
            }
        )

        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to get token")

        token_data = response.json()
        token_manager.save_token(service, token_data)

        return RedirectResponse(url=f"/?oauth_success={service}")
