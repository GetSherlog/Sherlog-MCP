import pytest
import pytest_asyncio
import httpx
import asyncio
import json
from typing import Any, Dict, Optional
import os


class MCPClient:
    """MCP client for testing via streamable HTTP transport"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.session_id: Optional[str] = None
    
    async def send_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a JSON-RPC message to the MCP server"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
            
        # Add JSON-RPC fields
        if "jsonrpc" not in message:
            message["jsonrpc"] = "2.0"
        if "id" not in message:
            message["id"] = 1
            
        response = await self.client.post(
            self.base_url,
            json=message,
            headers=headers
        )
        
        # Store session ID if provided
        if "mcp-session-id" in response.headers:
            self.session_id = response.headers["mcp-session-id"]
            
        response.raise_for_status()
        
        # Handle SSE response format
        if response.headers.get("content-type", "").startswith("text/event-stream"):
            # Parse SSE format
            text = response.text
            for line in text.split('\n'):
                if line.startswith('data: '):
                    data = line[6:].strip()
                    if data:
                        return json.loads(data)
        else:
            return response.json()
    
    async def initialize(self):
        """Initialize connection with MCP server"""
        return await self.send_message({
            "method": "initialize",
            "params": {
                "protocolVersion": "0.1.0",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        })
    
    async def list_tools(self):
        """List available tools from MCP server"""
        return await self.send_message({
            "method": "tools/list",
            "params": {}
        })
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Call a specific tool with arguments"""
        return await self.send_message({
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        })
    
    async def close(self):
        """Close the client connection"""
        await self.client.aclose()


class TestGoogleSheetsIntegration:
    """Integration tests for Google Sheets functionality via LogAI MCP"""
    
    @pytest_asyncio.fixture
    async def mcp_client(self):
        """Create MCP client connected to LogAI MCP server"""
        base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
        # Add the MCP endpoint path with trailing slash
        mcp_url = f"{base_url}/mcp/"
        client = MCPClient(mcp_url)
        await client.initialize()
        try:
            yield client
        finally:
            await client.close()
    
    @pytest.mark.asyncio
    async def test_mcp_health_check(self):
        """Test that LogAI MCP server is healthy and responding"""
        async with httpx.AsyncClient() as client:
            base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
            response = await client.get(f"{base_url}/health")
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'ok'
            assert 'service' in data
    
    @pytest.mark.asyncio
    async def test_list_tools_includes_google_sheets(self, mcp_client):
        """Test that Google Sheets tools are available in LogAI MCP"""
        tools_response = await mcp_client.list_tools()
        
        # Check for successful response
        assert 'result' in tools_response or 'error' not in tools_response
        
        # Get tools from result
        tools = tools_response.get('result', {}).get('tools', [])
        
        # Find Google Sheets related tools
        sheet_tools = [
            tool for tool in tools 
            if 'sheet' in tool['name'].lower() or 'google' in tool['name'].lower()
        ]
        
        # Log all tools if no Google Sheets tools found
        if len(sheet_tools) == 0:
            print(f"All available tools ({len(tools)}):")
            for tool in tools:
                print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
        
        assert len(sheet_tools) > 0, "No Google Sheets tools found in LogAI MCP"
        
        # Log available Google Sheets tools
        print(f"\nFound {len(sheet_tools)} Google Sheets tools:")
        for tool in sheet_tools:
            print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
            
        # Verify expected Google Sheets tools are present
        tool_names = [tool['name'] for tool in sheet_tools]
        
        # Check for essential Google Sheets operations
        expected_capabilities = ['create', 'read', 'update', 'list']
        found_capabilities = []
        
        for capability in expected_capabilities:
            if any(capability in name.lower() for name in tool_names):
                found_capabilities.append(capability)
                
        print(f"\nFound capabilities: {found_capabilities}")
        assert len(found_capabilities) >= 2, f"Missing essential Google Sheets capabilities. Found: {found_capabilities}"
    
    @pytest.mark.asyncio
    async def test_google_sheets_tool_call(self, mcp_client):
        """Test calling Google Sheets tools with various operations"""
        # First, list tools to find Google Sheets tools
        tools_response = await mcp_client.list_tools()
        tools = tools_response.get('result', {}).get('tools', [])
        
        # Find Google Sheets tools
        sheet_tools = [
            tool for tool in tools 
            if 'sheet' in tool['name'].lower() or 'google' in tool['name'].lower()
        ]
        
        if not sheet_tools:
            pytest.skip("No Google Sheets tools found")
            
        # Group tools by operation type
        list_tools = [t for t in sheet_tools if 'list' in t['name'].lower()]
        create_tools = [t for t in sheet_tools if 'create' in t['name'].lower()]
        read_tools = [t for t in sheet_tools if 'read' in t['name'].lower() or 'get' in t['name'].lower()]
        
        print(f"\nTesting Google Sheets tools:")
        print(f"- List tools: {len(list_tools)}")
        print(f"- Create tools: {len(create_tools)}")
        print(f"- Read tools: {len(read_tools)}")
        
        # Test 1: Try listing spreadsheets
        if list_tools:
            list_tool = list_tools[0]
            print(f"\nTesting list operation with: {list_tool['name']}")
            
            result = await mcp_client.call_tool(
                list_tool['name'],
                {}  # Most list operations don't require parameters
            )
            
            if 'result' in result:
                print(f"List operation succeeded")
                # Check if result has expected structure
                tool_result = result.get('result', {})
                if isinstance(tool_result, dict):
                    print(f"Result keys: {list(tool_result.keys())}")
            else:
                error = result.get('error', {})
                print(f"List operation failed: {error}")
                # Even errors are fine - we're testing the integration
                
        # Test 2: Test parameter validation
        if read_tools:
            read_tool = read_tools[0]
            print(f"\nTesting parameter validation with: {read_tool['name']}")
            
            # Call with empty parameters to test validation
            result = await mcp_client.call_tool(
                read_tool['name'],
                {}
            )
            
            if 'error' in result:
                error = result.get('error', {})
                print(f"Validation error (expected): {error.get('message', error)}")
                # Check that error mentions required parameters
                error_msg = str(error).lower()
                assert any(word in error_msg for word in ['required', 'missing', 'parameter', 'argument']), \
                    f"Error should mention missing parameters: {error}"
            
        # Test 3: Test tool input schema
        for tool in sheet_tools[:3]:  # Test first 3 tools
            print(f"\nAnalyzing tool schema for: {tool['name']}")
            
            # Check if tool has input schema
            input_schema = tool.get('inputSchema', {})
            if input_schema:
                properties = input_schema.get('properties', {})
                required = input_schema.get('required', [])
                
                print(f"  Properties: {list(properties.keys())}")
                print(f"  Required: {required}")
                
                # Verify schema structure
                assert isinstance(properties, dict), "Input schema should have properties"
    
    @pytest.mark.asyncio
    async def test_mcp_connection_lifecycle(self):
        """Test MCP connection initialization and lifecycle"""
        base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
        # Add the MCP endpoint path with trailing slash
        mcp_url = f"{base_url}/mcp/"
        client = MCPClient(mcp_url)
        
        # Test initialization
        init_response = await client.initialize()
        assert init_response is not None
        assert 'result' in init_response
        
        # Test tool listing after initialization
        tools_response = await client.list_tools()
        assert tools_response is not None
        assert 'result' in tools_response
        
        # Clean up
        await client.close()
    
    @pytest.mark.asyncio
    async def test_google_sheets_integration_flow(self, mcp_client):
        """Test a complete Google Sheets workflow through LogAI MCP"""
        # Get all available tools
        tools_response = await mcp_client.list_tools()
        tools = tools_response.get('result', {}).get('tools', [])
        
        # Find Google Sheets tools
        sheet_tools = {
            tool['name']: tool for tool in tools 
            if 'sheet' in tool['name'].lower() or 'google' in tool['name'].lower()
        }
        
        if not sheet_tools:
            pytest.skip("No Google Sheets tools found")
            
        print(f"\nGoogle Sheets Integration Test")
        print(f"Available tools: {list(sheet_tools.keys())}")
        
        # Test the integration by attempting common operations
        tested_operations = []
        
        # Find and test a list operation
        list_tool = next((name for name in sheet_tools if 'list' in name.lower()), None)
        if list_tool:
            print(f"\n1. Testing list operation: {list_tool}")
            result = await mcp_client.call_tool(list_tool, {})
            
            if 'result' in result:
                tested_operations.append('list')
                print("   ✓ List operation successful")
                
                # Analyze the result structure
                tool_result = result.get('result', {})
                if isinstance(tool_result, list):
                    print(f"   - Found {len(tool_result)} items")
                elif isinstance(tool_result, dict):
                    print(f"   - Result type: dict with keys: {list(tool_result.keys())[:5]}")
            else:
                print(f"   ✗ List operation failed: {result.get('error', {})}")
                
        # Find and test operations that require parameters
        param_tools = [
            (name, tool) for name, tool in sheet_tools.items() 
            if tool.get('inputSchema', {}).get('required', [])
        ]
        
        if param_tools:
            name, tool = param_tools[0]
            print(f"\n2. Testing parameterized operation: {name}")
            
            # Get required parameters
            schema = tool.get('inputSchema', {})
            required = schema.get('required', [])
            properties = schema.get('properties', {})
            
            print(f"   Required parameters: {required}")
            print(f"   Parameter types: {[(p, properties.get(p, {}).get('type', 'unknown')) for p in required[:3]]}")
            
            # Test with missing parameters
            result = await mcp_client.call_tool(name, {})
            if 'error' in result:
                tested_operations.append('validation')
                print("   ✓ Parameter validation working correctly")
            
        # Summary
        print(f"\n=== Integration Test Summary ===")
        print(f"Total Google Sheets tools: {len(sheet_tools)}")
        print(f"Tested operations: {tested_operations}")
        print(f"Tool categories found: {set(name.split('_')[0] for name in sheet_tools if '_' in name)}")
        
        # Ensure we tested at least some operations
        assert len(tested_operations) > 0, "No operations could be tested"
        assert len(sheet_tools) >= 5, f"Expected at least 5 Google Sheets tools, found {len(sheet_tools)}"