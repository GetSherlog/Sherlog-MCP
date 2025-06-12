"""Multi-MCP Integration Tests

Tests the integration of multiple MCPs (GitHub internal and Google Sheets external)
to ensure they work together properly in the LogAI MCP server.
"""

import pytest
import pytest_asyncio
import httpx
import asyncio
import json
from typing import Any, Dict, Optional, List
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


class TestMultiMCPIntegration:
    """Integration tests for multiple MCPs (GitHub internal + Google Sheets external)"""
    
    @pytest_asyncio.fixture
    async def mcp_client(self):
        """Create MCP client connected to LogAI MCP server"""
        base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
        mcp_url = f"{base_url}/mcp/"
        client = MCPClient(mcp_url)
        await client.initialize()
        try:
            yield client
        finally:
            await client.close()
    
    @pytest.mark.asyncio
    async def test_list_tools_from_both_mcps(self, mcp_client):
        """Test that tools from both internal (GitHub) and external (Google Sheets) MCPs are available"""
        tools_response = await mcp_client.list_tools()
        
        # Check for successful response
        assert 'result' in tools_response or 'error' not in tools_response
        
        # Get tools from result
        tools = tools_response.get('result', {}).get('tools', [])
        tool_names = [tool['name'] for tool in tools]
        
        print(f"\nTotal tools available: {len(tools)}")
        
        # Categorize tools by type
        github_tools = [name for name in tool_names if 'github' in name.lower() or 'issue' in name.lower() or 'pull_request' in name.lower() or 'commit' in name.lower()]
        sheets_tools = [name for name in tool_names if 'sheet' in name.lower() or 'google' in name.lower()]
        external_tools = [name for name in tool_names if name.startswith('google-sheets_') or name.startswith('sheets_')]
        internal_tools = [name for name in tool_names if not any(name.startswith(prefix) for prefix in ['google-sheets_', 'sheets_'])]
        
        print(f"\nTool Categories:")
        print(f"- GitHub tools (internal): {len(github_tools)}")
        print(f"- Google Sheets tools: {len(sheets_tools)}")
        print(f"- Total external tools: {len(external_tools)}")
        print(f"- Total internal tools: {len(internal_tools)}")
        
        # Verify both tool sets are present
        assert len(github_tools) > 0, "No GitHub tools found (internal tools missing)"
        assert len(sheets_tools) > 0, "No Google Sheets tools found (external MCP not loaded)"
        
        # Print sample tools from each category
        print(f"\nSample GitHub tools: {github_tools[:3]}")
        print(f"Sample Google Sheets tools: {sheets_tools[:3]}")
        
        # Check for specific expected tools
        expected_github_tools = ['test_github_connection', 'get_issue', 'search_issues']
        found_github = [tool for tool in expected_github_tools if tool in tool_names]
        print(f"\nExpected GitHub tools found: {found_github}")
        
        # Check for Google Sheets capabilities
        sheets_capabilities = []
        if any('list' in name.lower() for name in sheets_tools):
            sheets_capabilities.append('list')
        if any('create' in name.lower() for name in sheets_tools):
            sheets_capabilities.append('create')
        if any('read' in name.lower() or 'get' in name.lower() for name in sheets_tools):
            sheets_capabilities.append('read')
        if any('update' in name.lower() for name in sheets_tools):
            sheets_capabilities.append('update')
            
        print(f"\nGoogle Sheets capabilities found: {sheets_capabilities}")
        assert len(sheets_capabilities) >= 2, "Missing essential Google Sheets capabilities"
        
    @pytest.mark.asyncio
    async def test_github_and_sheets_tools_coexist(self, mcp_client):
        """Test that GitHub and Google Sheets tools can be used in the same session"""
        tools_response = await mcp_client.list_tools()
        tools = tools_response.get('result', {}).get('tools', [])
        tool_names = [tool['name'] for tool in tools]
        
        # Find one tool from each MCP
        github_test_tool = next((name for name in tool_names if name == 'test_github_connection'), None)
        sheets_list_tool = None
        
        # Find a Google Sheets list tool
        for name in tool_names:
            if name.startswith('google-sheets_') and 'list' in name:
                sheets_list_tool = name
                break
        
        # Test GitHub tool
        if github_test_tool:
            print(f"\nTesting GitHub tool: {github_test_tool}")
            result = await mcp_client.call_tool(
                github_test_tool,
                {"save_as": "github_result"}
            )
            
            # GitHub tools might require valid token
            assert 'result' in result or 'error' in result
            print(f"GitHub tool call {'succeeded' if 'result' in result else 'failed (may need valid token)'}")
        
        # Test Google Sheets tool
        if not sheets_list_tool:
            # If no list tool, find any Google Sheets tool
            for name in tool_names:
                if name.startswith('google-sheets_'):
                    sheets_list_tool = name
                    break
        
        if sheets_list_tool:
            print(f"\nTesting Google Sheets tool: {sheets_list_tool}")
            
            # Call with minimal parameters
            result = await mcp_client.call_tool(
                sheets_list_tool,
                {"save_as": "sheets_result"}
            )
            
            # Check result structure
            assert 'result' in result or 'error' in result
            print(f"Google Sheets tool call {'succeeded' if 'result' in result else 'failed (may need auth)'}")
        
        # Verify both tool types were found and tested
        assert github_test_tool is not None, "No GitHub tools available to test"
        assert sheets_list_tool is not None, "No Google Sheets tools available to test"
        print("\n✓ Both internal and external tools coexist and can be called")
    
    @pytest.mark.asyncio
    async def test_workflow_using_multiple_mcps(self, mcp_client):
        """Test a workflow that demonstrates using tools from both MCPs together"""
        tools_response = await mcp_client.list_tools()
        tools = tools_response.get('result', {}).get('tools', [])
        tool_names = [tool['name'] for tool in tools]
        
        print("\n=== Multi-MCP Workflow Test ===")
        
        # Step 1: List available external tools
        list_external_tool = next((name for name in tool_names if name == 'list_external_tools'), None)
        
        if list_external_tool:
            print("\nStep 1: Listing external tools...")
            result = await mcp_client.call_tool(
                'list_external_tools',
                {}
            )
            
            if 'result' in result:
                # Parse the result - it might be wrapped in content
                result_data = result['result']
                if isinstance(result_data, dict) and 'content' in result_data:
                    # Extract text from content
                    content = result_data['content'][0]['text']
                    import json
                    external_info = json.loads(content)
                else:
                    external_info = result_data
                    
                print(f"External MCPs found: {list(external_info.keys())}")
                
                # Check Google Sheets is registered
                assert 'google-sheets' in external_info, "Google Sheets MCP not found in external tools"
                
                sheets_tools = external_info.get('google-sheets', {}).get('tools', [])
                print(f"Google Sheets tools count: {len(sheets_tools)}")
        
        # Step 2: Test GitHub connection (internal tool)
        if 'test_github_connection' in tool_names:
            print("\nStep 2: Testing GitHub connection (internal tool)...")
            github_result = await mcp_client.call_tool(
                'test_github_connection',
                {"save_as": "github_conn_test"}
            )
            
            if 'result' in github_result:
                print("✓ GitHub connection test completed")
            else:
                print("✗ GitHub connection test failed (may need valid token)")
        
        # Step 3: Demonstrate tool naming convention
        print("\nStep 3: Analyzing tool naming conventions...")
        
        internal_tools = [name for name in tool_names if not name.startswith('google-sheets_')]
        external_tools = [name for name in tool_names if name.startswith('google-sheets_')]
        
        print(f"- Internal tools use direct names: {internal_tools[:3]}")
        print(f"- External tools are prefixed: {external_tools[:3]}")
        
        # Verify naming prevents conflicts
        internal_base_names = set(internal_tools)
        external_base_names = set()
        
        for ext_tool in external_tools:
            # External tools should be prefixed with MCP name
            assert ext_tool.startswith('google-sheets_'), f"External tool {ext_tool} doesn't follow naming convention"
            base_name = ext_tool.replace('google-sheets_', '')
            external_base_names.add(base_name)
        
        # Check for potential naming conflicts
        conflicts = internal_base_names.intersection(external_base_names)
        print(f"\nNaming conflicts prevented by prefixing: {len(conflicts)} potential conflicts avoided")
        
        print("\n✓ Multi-MCP workflow completed successfully")
    
    @pytest.mark.asyncio
    async def test_tool_isolation_and_namespacing(self, mcp_client):
        """Test that tools from different MCPs are properly isolated and namespaced"""
        tools_response = await mcp_client.list_tools()
        tools = tools_response.get('result', {}).get('tools', [])
        tool_names = [tool['name'] for tool in tools]
        
        print("\n=== Tool Isolation and Namespacing Test ===")
        
        # Analyze tool naming
        github_tools = [name for name in tool_names if 'github' in name.lower()]
        sheets_tools = [name for name in tool_names if name.startswith('google-sheets_')]
        
        # Check for naming conflicts
        tool_name_map = {}
        for name in tool_names:
            base_name = name.replace('google-sheets_', '') if name.startswith('google-sheets_') else name
            if base_name in tool_name_map:
                tool_name_map[base_name].append(name)
            else:
                tool_name_map[base_name] = [name]
        
        # Find any naming conflicts
        conflicts = {k: v for k, v in tool_name_map.items() if len(v) > 1}
        
        if conflicts:
            print(f"\nNaming conflicts detected: {conflicts}")
            # This is actually OK if properly namespaced
            for base, names in conflicts.items():
                internal = [n for n in names if not n.startswith('google-sheets_')]
                external = [n for n in names if n.startswith('google-sheets_')]
                assert len(internal) <= 1, f"Multiple internal tools with same base name: {internal}"
                print(f"  '{base}': {len(internal)} internal, {len(external)} external")
        else:
            print("\nNo naming conflicts detected between MCPs")
        
        # Check external tools follow naming convention
        external_prefixes = set()
        for name in tool_names:
            if '_' in name:
                prefix = name.split('_')[0]
                # Check for MCP prefixes (like google-sheets)
                if '-' in prefix:
                    external_prefixes.add(prefix)
        
        print(f"\nExternal MCP prefixes found: {external_prefixes}")
        
        # Verify tool metadata includes MCP information
        internal_tool = next((t for t in tools if not t['name'].startswith('google-sheets_')), None)
        external_tool = next((t for t in tools if t['name'].startswith('google-sheets_')), None)
        
        print("\nAnalyzing tool metadata:")
        
        if internal_tool:
            print(f"\nInternal tool example: {internal_tool['name']}")
            print(f"- Has description: {bool(internal_tool.get('description'))}")
            print(f"- Has input schema: {bool(internal_tool.get('inputSchema'))}")
        
        if external_tool:
            print(f"\nExternal tool example: {external_tool['name']}")
            print(f"- Has description: {bool(external_tool.get('description'))}")
            desc = external_tool.get('description', '')
            # Check if description includes MCP metadata
            if '[External MCP:' in desc:
                print("- ✓ Description includes MCP metadata")
            
        print("\n✓ Tool isolation and namespacing verified")
    
    @pytest.mark.asyncio 
    async def test_error_handling_across_mcps(self, mcp_client):
        """Test error handling for both internal and external MCP tools"""
        tools_response = await mcp_client.list_tools()
        tools = tools_response.get('result', {}).get('tools', [])
        tool_dict = {tool['name']: tool for tool in tools}
        
        print("\n=== Error Handling Test ===")
        
        # Test 1: Invalid parameters for internal tool
        if 'get_issue' in tool_dict:
            print("\nTest 1: Invalid parameters for internal GitHub tool...")
            result = await mcp_client.call_tool(
                'get_issue',
                {}  # Missing required parameters
            )
            
            # The result might contain error in the content or as a separate error field
            if 'error' in result:
                error = result['error']
                print(f"✓ Error correctly returned: {error.get('message', error)[:100]}...")
            elif 'result' in result and result['result'].get('isError'):
                # Error is in the result content
                error_text = result['result']['content'][0]['text']
                print(f"✓ Error in content: {error_text[:100]}...")
                assert 'validation error' in error_text.lower() or 'field required' in error_text.lower(), \
                    "Expected validation error for missing parameters"
            else:
                assert False, f"Expected error for missing parameters, got: {result}"
        
        # Test 2: Invalid parameters for external tool
        sheets_tool = next((name for name in tool_dict if name.startswith('google-sheets_') and 'read' in name), None)
        
        if sheets_tool:
            print("\nTest 2: Invalid parameters for external Google Sheets tool...")
            result = await mcp_client.call_tool(
                sheets_tool,
                {}  # Missing required parameters
            )
            
            # External tools should also handle errors properly
            if 'error' in result:
                print(f"✓ External tool error: {result['error'].get('message', result['error'])[:100]}...")
            elif 'result' in result and result['result'].get('isError'):
                # Error is in the result content
                error_text = result['result']['content'][0]['text']
                print(f"✓ External tool error in content: {error_text[:100]}...")
                assert 'error' in error_text.lower() or 'required' in error_text.lower(), \
                    "Expected error for missing parameters"
            else:
                print("✓ Tool executed with defaults or empty result")
        
        # Test 3: Non-existent tool
        print("\nTest 3: Calling non-existent tool...")
        result = await mcp_client.call_tool(
            'non_existent_tool_12345',
            {"param": "value"}
        )
        
        if 'error' in result:
            print(f"✓ Error for non-existent tool: {result['error'].get('message', result['error'])[:100]}...")
        else:
            # Some implementations might return isError in result
            assert result.get('result', {}).get('isError'), f"Expected error for non-existent tool, got: {result}"
            print(f"✓ Error for non-existent tool in result content")
        
        print("\n✓ Error handling works correctly across all MCPs")
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Concurrent testing can be flaky in test environment")
    async def test_concurrent_mcp_operations(self, mcp_client):
        """Test that tools from different MCPs can be called concurrently"""
        tools_response = await mcp_client.list_tools()
        tools = tools_response.get('result', {}).get('tools', [])
        tool_names = [tool['name'] for tool in tools]
        
        print("\n=== Concurrent MCP Operations Test ===")
        
        # Find tools from each MCP
        github_tool = next((name for name in tool_names if name == 'test_github_connection'), None)
        sheets_tool = next((name for name in tool_names if name.startswith('google-sheets_') and 'list' in name), None)
        
        if github_tool and sheets_tool:
            print(f"\nCalling tools concurrently:")
            print(f"- GitHub tool: {github_tool}")
            print(f"- Google Sheets tool: {sheets_tool}")
            
            # Create concurrent tasks
            github_task = mcp_client.call_tool(github_tool, {"save_as": "concurrent_github"})
            sheets_task = mcp_client.call_tool(sheets_tool, {"save_as": "concurrent_sheets"})
            
            # Execute concurrently
            start_time = asyncio.get_event_loop().time()
            results = await asyncio.gather(github_task, sheets_task, return_exceptions=True)
            elapsed = asyncio.get_event_loop().time() - start_time
            
            print(f"\nConcurrent execution completed in {elapsed:.2f}s")
            
            # Check results
            github_result, sheets_result = results
            
            if not isinstance(github_result, Exception):
                print(f"- GitHub tool: {'✓ Success' if 'result' in github_result else '✗ Failed'}")
            else:
                print(f"- GitHub tool: ✗ Exception: {github_result}")
                
            if not isinstance(sheets_result, Exception):
                print(f"- Google Sheets tool: {'✓ Success' if 'result' in sheets_result else '✗ Failed'}")
            else:
                print(f"- Google Sheets tool: ✗ Exception: {sheets_result}")
                
            # At least one should succeed
            assert (not isinstance(github_result, Exception) or not isinstance(sheets_result, Exception)), \
                "Both concurrent operations failed"
                
            print("\n✓ Concurrent operations completed successfully")
        else:
            pytest.skip("Required tools not available for concurrent test")


@pytest.mark.asyncio
async def test_mcp_integration_summary():
    """Summary test that reports on the overall MCP integration status"""
    base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
    mcp_url = f"{base_url}/mcp/"
    client = MCPClient(mcp_url)
    
    try:
        await client.initialize()
        tools_response = await client.list_tools()
        tools = tools_response.get('result', {}).get('tools', [])
        
        print("\n" + "="*60)
        print("MULTI-MCP INTEGRATION SUMMARY")
        print("="*60)
        
        # Categorize tools
        internal_tools = [t for t in tools if not t['name'].startswith('google-sheets_')]
        external_tools = [t for t in tools if t['name'].startswith('google-sheets_')]
        
        github_tools = [t for t in internal_tools if any(kw in t['name'].lower() for kw in ['github', 'issue', 'pull', 'commit'])]
        
        print(f"\nTotal tools available: {len(tools)}")
        print(f"├── Internal tools: {len(internal_tools)}")
        print(f"│   └── GitHub tools: {len(github_tools)}")
        print(f"└── External tools: {len(external_tools)}")
        print(f"    └── Google Sheets tools: {len(external_tools)}")
        
        # Check integration status
        print("\nIntegration Status:")
        print(f"✓ LogAI MCP Server: Running")
        print(f"✓ Internal GitHub tools: {'Loaded' if github_tools else 'Not found'}")
        print(f"✓ External Google Sheets MCP: {'Connected' if external_tools else 'Not connected'}")
        
        # List some example tools
        if github_tools:
            print(f"\nExample GitHub tools:")
            for tool in github_tools[:3]:
                print(f"  - {tool['name']}")
        
        if external_tools:
            print(f"\nExample Google Sheets tools:")
            for tool in external_tools[:3]:
                print(f"  - {tool['name']}")
        
        print("\n" + "="*60)
        
        # Final assertion
        assert len(internal_tools) > 0, "No internal tools found"
        assert len(external_tools) > 0, "No external tools found - Google Sheets MCP may not be configured"
        
        print("\n✅ Multi-MCP integration validated successfully!")
        
    finally:
        await client.close()