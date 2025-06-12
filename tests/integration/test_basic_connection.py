import pytest
import httpx
import os
import json


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health endpoint is accessible"""
    base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'


@pytest.mark.asyncio 
async def test_mcp_endpoint_exists():
    """Test that MCP endpoint exists"""
    base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
    async with httpx.AsyncClient() as client:
        # Try different endpoints
        endpoints = [
            "/mcp",
            "/mcp/", 
            "/",
            "/sse"
        ]
        
        for endpoint in endpoints:
            response = await client.get(f"{base_url}{endpoint}")
            print(f"GET {endpoint}: {response.status_code}")
            
        # Try POST to MCP endpoint with different content types
        for content_type in ["application/json", "text/event-stream", "application/x-ndjson"]:
            headers = {"Content-Type": content_type, "Accept": content_type}
            response = await client.post(
                f"{base_url}/mcp/",
                json={"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1},
                headers=headers
            )
            print(f"POST /mcp/ with {content_type}: {response.status_code}")
            if response.status_code == 200:
                print(f"Success with {content_type}!")
                print(f"Response: {response.text}")
                break


@pytest.mark.asyncio
async def test_list_available_endpoints():
    """List all available endpoints on the server"""
    base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
    async with httpx.AsyncClient() as client:
        # Try to get the root to see if it lists endpoints
        response = await client.get(base_url)
        print(f"Root response: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        if response.status_code == 200:
            print(f"Content: {response.text[:500]}...")  # First 500 chars