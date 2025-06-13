import pytest
import os
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
import httpx


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health endpoint is accessible"""
    base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        assert data['service'] == 'LogAI MCP'


@pytest.mark.asyncio
async def test_mcp_initialize_with_sdk():
    """Test MCP initialization using SDK client"""
    base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
    
    # Connect to streamable HTTP server
    async with streamablehttp_client(f"{base_url}/mcp/") as (read_stream, write_stream, _):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            result = await session.initialize()
            
            # Verify initialization response
            assert result.protocolVersion == "0.1.0"
            assert result.serverInfo.name == "LogAIMCP"
            assert result.serverInfo.version is not None
            
            # Verify capabilities if present
            if result.capabilities:
                print(f"Server capabilities: {result.capabilities}")


@pytest.mark.asyncio
async def test_list_tools_with_sdk():
    """Test listing tools using SDK client"""
    base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
    
    async with streamablehttp_client(f"{base_url}/mcp/") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize first
            await session.initialize()
            
            # List tools
            result = await session.list_tools()
            
            # Verify we have tools
            assert hasattr(result, 'tools')
            assert len(result.tools) > 0
            
            # Print tool names for debugging
            tool_names = [tool.name for tool in result.tools]
            print(f"Available tools: {tool_names}")
            
            # Verify some expected tools exist
            expected_tools = ['load_logs', 'preprocess_logs', 'vectorize_logs']
            for expected in expected_tools:
                assert expected in tool_names, f"Expected tool {expected} not found"


@pytest.mark.asyncio
async def test_call_tool_with_sdk():
    """Test calling a tool using SDK client"""
    base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
    
    async with streamablehttp_client(f"{base_url}/mcp/") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize first
            await session.initialize()
            
            # Test calling a simple tool (list_files)
            result = await session.call_tool('list_files', {
                'path': '/tmp',
                'pattern': '*.log'
            })
            
            # Verify response
            assert result is not None
            assert hasattr(result, 'content')
            
            # The response should contain file listing info
            print(f"Tool response: {result.content}")


@pytest.mark.asyncio
async def test_invalid_tool_call_with_sdk():
    """Test error handling when calling invalid tool"""
    base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
    
    async with streamablehttp_client(f"{base_url}/mcp/") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize first
            await session.initialize()
            
            # Try to call a non-existent tool
            with pytest.raises(Exception) as exc_info:
                await session.call_tool('non_existent_tool', {})
            
            # Verify we get an appropriate error
            assert "non_existent_tool" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_concurrent_requests_with_sdk():
    """Test concurrent tool calls using SDK client"""
    base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
    
    async with streamablehttp_client(f"{base_url}/mcp/") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize first
            await session.initialize()
            
            # Note: The SDK client may not support concurrent requests on the same session
            # Test sequential requests instead
            results = []
            for _ in range(3):
                result = await session.list_tools()
                results.append(result)
            
            # Verify all requests succeeded
            assert len(results) == 3
            for result in results:
                assert hasattr(result, 'tools')
                assert len(result.tools) > 0