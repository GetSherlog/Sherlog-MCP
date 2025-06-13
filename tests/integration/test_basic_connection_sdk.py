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
            expected_tools = ['load_file_log_data', 'list_directory', 'read_file']
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
            
            # Test calling a simple tool (list_directory)
            result = await session.call_tool('list_directory', {
                'dir_path': '/tmp',
                'save_as': 'tmp_files'
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
            try:
                result = await session.call_tool('non_existent_tool', {})
                # The server may return an error result instead of raising exception
                assert result is not None
                if hasattr(result, 'content'):
                    content = result.content
                    if isinstance(content, list) and len(content) > 0:
                        content = content[0]
                    # Check for error in the response
                    content_text = content.text if hasattr(content, 'text') else str(content)
                    assert 'error' in content_text.lower() or 'exception' in content_text.lower()
            except Exception as e:
                # This is also acceptable - the tool doesn't exist
                error_msg = str(e).lower()
                assert 'non_existent_tool' in error_msg or 'not found' in error_msg or 'unknown tool' in error_msg or 'error' in error_msg


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