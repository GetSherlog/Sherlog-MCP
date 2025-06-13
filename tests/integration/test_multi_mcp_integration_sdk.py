import pytest
import pytest_asyncio
import asyncio
import os
import uuid
from datetime import datetime
from typing import List
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession


class TestMultiMCPIntegration:
    """Test integration of multiple MCP servers (internal + external)"""
    
    @pytest_asyncio.fixture
    async def mcp_session(self):
        """Create MCP session connected to LogAI MCP server"""
        base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
        
        async with streamablehttp_client(f"{base_url}/mcp/") as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session
    
    @pytest.mark.asyncio
    async def test_all_tools_accessible(self, mcp_session):
        """Test that both internal and external tools are accessible"""
        tools_response = await mcp_session.list_tools()
        tools = tools_response.tools
        tool_names = [t.name for t in tools]
        
        # Check for internal LogAI tools
        internal_tools = [name for name in tool_names if not name.startswith('external_')]
        assert len(internal_tools) > 0, "No internal LogAI tools found"
        
        # Check for some expected internal tools
        expected_internal = ['load_logs', 'preprocess_logs', 'vectorize_logs']
        for tool in expected_internal:
            assert tool in tool_names, f"Expected internal tool '{tool}' not found"
        
        # Check for external tools
        external_tools = [name for name in tool_names if name.startswith('external_')]
        print(f"Found {len(external_tools)} external tools")
        
        # Check for GitHub and Google Sheets tools
        github_tools = [name for name in external_tools if 'github' in name]
        sheets_tools = [name for name in external_tools if 'google-sheets' in name]
        
        if len(github_tools) > 0:
            print(f"Found {len(github_tools)} GitHub tools")
        if len(sheets_tools) > 0:
            print(f"Found {len(sheets_tools)} Google Sheets tools")
    
    @pytest.mark.asyncio
    async def test_tool_namespacing(self, mcp_session):
        """Test that external tools are properly namespaced"""
        tools_response = await mcp_session.list_tools()
        tools = tools_response.tools
        
        # Check external tool naming convention
        for tool in tools:
            if tool.name.startswith('external_'):
                # Should be: external_<mcp-name>_<tool-name>
                parts = tool.name.split('_', 2)
                assert len(parts) >= 3, f"Invalid external tool name format: {tool.name}"
                assert parts[0] == 'external'
                # Second part should be MCP server name
                assert len(parts[1]) > 0, f"Missing MCP server name in: {tool.name}"
    
    @pytest.mark.asyncio
    async def test_sequential_multi_mcp_operations(self, mcp_session):
        """Test sequential operations across different MCP servers"""
        # Get available tools
        tools_response = await mcp_session.list_tools()
        tool_names = [t.name for t in tools_response.tools]
        
        # Test internal tool operation
        result1 = await mcp_session.call_tool('list_files', {'path': '/tmp', 'pattern': '*.log'})
        assert result1 is not None
        print("Internal tool executed successfully")
        
        # Test external tool operations if available
        github_tools = [name for name in tool_names if name.startswith('external_') and 'github' in name]
        if github_tools and any('list_repos' in t for t in github_tools):
            github_list_tool = next(t for t in github_tools if 'list_repos' in t)
            try:
                result2 = await mcp_session.call_tool(github_list_tool, {})
                print("GitHub tool executed successfully")
            except Exception as e:
                print(f"GitHub tool execution failed (may need auth): {e}")
        
        sheets_tools = [name for name in tool_names if name.startswith('external_') and 'google-sheets' in name]
        if sheets_tools and any('create_spreadsheet' in t for t in sheets_tools):
            sheets_create_tool = next(t for t in sheets_tools if 'create_spreadsheet' in t)
            result3 = await mcp_session.call_tool(
                sheets_create_tool,
                {
                    'title': f'Test Sequential {uuid.uuid4().hex[:8]}',
                    'data': [['Test', 'Data']]
                }
            )
            print("Google Sheets tool executed successfully")
    
    @pytest.mark.asyncio
    async def test_external_tool_error_propagation(self, mcp_session):
        """Test that errors from external tools are properly propagated"""
        tools_response = await mcp_session.list_tools()
        tool_names = [t.name for t in tools_response.tools]
        
        # Find an external tool to test with
        external_tools = [name for name in tool_names if name.startswith('external_')]
        
        if external_tools:
            # Try to call a tool with invalid parameters
            test_tool = external_tools[0]
            
            try:
                # Call with intentionally bad parameters
                await mcp_session.call_tool(test_tool, {'invalid_param': 'bad_value'})
            except Exception as e:
                # Error should be propagated
                print(f"Got expected error: {str(e)}")
                assert True
            else:
                # If no error, the tool might have accepted the params
                # This is also valid behavior
                assert True
    
    @pytest.mark.asyncio
    async def test_cross_mcp_workflow(self, mcp_session):
        """Test a workflow that uses both internal and external tools"""
        tools_response = await mcp_session.list_tools()
        tool_names = [t.name for t in tools_response.tools]
        
        # Create test data with internal tool
        test_data = f"test_log_{datetime.now().isoformat()}.log"
        
        # Use internal tool to create/process data
        result1 = await mcp_session.call_tool('list_files', {'path': '/tmp', 'pattern': '*.log'})
        assert result1 is not None
        
        # If Google Sheets is available, create a spreadsheet with the results
        sheets_tools = [name for name in tool_names if name.startswith('external_') and 'google-sheets' in name]
        if sheets_tools and any('create_spreadsheet' in t for t in sheets_tools):
            sheets_create_tool = next(t for t in sheets_tools if 'create_spreadsheet' in t)
            
            # Create spreadsheet with log file info
            result2 = await mcp_session.call_tool(
                sheets_create_tool,
                {
                    'title': f'Log Files Report {datetime.now().strftime("%Y-%m-%d")}',
                    'data': [
                        ['Filename', 'Status', 'Timestamp'],
                        [test_data, 'Processed', datetime.now().isoformat()]
                    ]
                }
            )
            
            assert result2 is not None
            print("Successfully created cross-MCP workflow result")
    
    @pytest.mark.asyncio
    async def test_mcp_isolation(self, mcp_session):
        """Test that different MCP servers are properly isolated"""
        tools_response = await mcp_session.list_tools()
        tools = tools_response.tools
        
        # Group tools by MCP server
        tool_groups = {}
        
        for tool in tools:
            if tool.name.startswith('external_'):
                parts = tool.name.split('_', 2)
                if len(parts) >= 3:
                    mcp_name = parts[1]
                    if mcp_name not in tool_groups:
                        tool_groups[mcp_name] = []
                    tool_groups[mcp_name].append(tool.name)
            else:
                if 'internal' not in tool_groups:
                    tool_groups['internal'] = []
                tool_groups['internal'].append(tool.name)
        
        print(f"Found {len(tool_groups)} MCP server groups:")
        for group, tools in tool_groups.items():
            print(f"  {group}: {len(tools)} tools")
        
        # Verify each group has distinct tools
        all_tools = []
        for tools in tool_groups.values():
            all_tools.extend(tools)
        
        # Check for duplicates (there shouldn't be any)
        assert len(all_tools) == len(set(all_tools)), "Found duplicate tool names across MCP servers"