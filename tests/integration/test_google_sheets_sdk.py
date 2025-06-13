import pytest
import pytest_asyncio
import os
import uuid
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession


class TestGoogleSheetsIntegration:
    """Integration tests for Google Sheets functionality via LogAI MCP"""
    
    @pytest_asyncio.fixture
    async def mcp_session(self):
        """Create MCP session connected to LogAI MCP server"""
        base_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8000')
        
        async with streamablehttp_client(f"{base_url}/mcp/") as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session
    
    @pytest.mark.asyncio
    async def test_google_sheets_tools_available(self, mcp_session):
        """Test that Google Sheets tools are available through LogAI MCP"""
        tools_response = await mcp_session.list_tools()
        
        # Check if we have tools
        assert hasattr(tools_response, 'tools')
        tools = tools_response.tools
        tool_names = [t.name for t in tools]
        
        # Check for Google Sheets tools with external prefix
        google_sheets_tools = [name for name in tool_names if name.startswith('external_google-sheets_')]
        print(f"Found Google Sheets tools: {google_sheets_tools}")
        
        # We should have some Google Sheets tools
        assert len(google_sheets_tools) > 0, "No Google Sheets tools found"
        
        # Check for expected tools
        expected_patterns = ['create_spreadsheet', 'read_spreadsheet', 'update_cell']
        for pattern in expected_patterns:
            matching = [t for t in google_sheets_tools if pattern in t]
            assert len(matching) > 0, f"No tool matching pattern '{pattern}' found"
    
    @pytest.mark.asyncio
    async def test_create_and_read_spreadsheet(self, mcp_session):
        """Test creating a spreadsheet and reading it back"""
        test_id = str(uuid.uuid4())[:8]
        
        # Create a test spreadsheet
        create_result = await mcp_session.call_tool(
            'external_google-sheets_create_spreadsheet',
            {
                'title': f'Test Spreadsheet {test_id}',
                'data': [
                    ['Name', 'Age', 'Email'],
                    ['Alice', '30', 'alice@example.com'],
                    ['Bob', '25', 'bob@example.com']
                ]
            }
        )
        
        # Check creation result
        assert create_result is not None
        assert hasattr(create_result, 'content')
        
        # Extract spreadsheet ID from result
        content = create_result.content
        if isinstance(content, list):
            content = content[0]
        
        result_text = content.text if hasattr(content, 'text') else str(content)
        print(f"Create result: {result_text}")
        
        # The result should contain the spreadsheet ID
        assert 'spreadsheet' in result_text.lower()
        
        # Extract ID (usually in format: "Created spreadsheet: [ID]")
        import re
        id_match = re.search(r'spreadsheet[:\s]+([a-zA-Z0-9_-]+)', result_text, re.IGNORECASE)
        if id_match:
            spreadsheet_id = id_match.group(1)
            print(f"Created spreadsheet ID: {spreadsheet_id}")
            
            # Try to read the spreadsheet
            read_result = await mcp_session.call_tool(
                'external_google-sheets_read_spreadsheet',
                {
                    'spreadsheet_id': spreadsheet_id,
                    'range': 'A1:C3'
                }
            )
            
            assert read_result is not None
            print(f"Read result: {read_result}")
    
    @pytest.mark.asyncio
    async def test_update_spreadsheet_cell(self, mcp_session):
        """Test updating a cell in a spreadsheet"""
        test_id = str(uuid.uuid4())[:8]
        
        # First create a spreadsheet
        create_result = await mcp_session.call_tool(
            'external_google-sheets_create_spreadsheet',
            {
                'title': f'Test Update {test_id}',
                'data': [['Original Value']]
            }
        )
        
        # Extract spreadsheet ID
        content = create_result.content
        if isinstance(content, list):
            content = content[0]
        
        result_text = content.text if hasattr(content, 'text') else str(content)
        
        import re
        id_match = re.search(r'spreadsheet[:\s]+([a-zA-Z0-9_-]+)', result_text, re.IGNORECASE)
        if id_match:
            spreadsheet_id = id_match.group(1)
            
            # Update the cell
            update_result = await mcp_session.call_tool(
                'external_google-sheets_update_cell',
                {
                    'spreadsheet_id': spreadsheet_id,
                    'range': 'A1',
                    'value': 'Updated Value'
                }
            )
            
            assert update_result is not None
            print(f"Update result: {update_result}")
            
            # Read back to verify
            read_result = await mcp_session.call_tool(
                'external_google-sheets_read_spreadsheet',
                {
                    'spreadsheet_id': spreadsheet_id,
                    'range': 'A1'
                }
            )
            
            assert read_result is not None
            result_content = read_result.content
            if isinstance(result_content, list):
                result_content = result_content[0]
            
            result_text = result_content.text if hasattr(result_content, 'text') else str(result_content)
            assert 'Updated Value' in result_text
    
    @pytest.mark.asyncio
    async def test_batch_update_spreadsheet(self, mcp_session):
        """Test batch updating multiple cells"""
        test_id = str(uuid.uuid4())[:8]
        
        # Create a spreadsheet with initial data
        create_result = await mcp_session.call_tool(
            'external_google-sheets_create_spreadsheet',
            {
                'title': f'Test Batch Update {test_id}',
                'data': [
                    ['A1', 'B1', 'C1'],
                    ['A2', 'B2', 'C2']
                ]
            }
        )
        
        # Extract spreadsheet ID
        content = create_result.content
        if isinstance(content, list):
            content = content[0]
        
        result_text = content.text if hasattr(content, 'text') else str(content)
        
        import re
        id_match = re.search(r'spreadsheet[:\s]+([a-zA-Z0-9_-]+)', result_text, re.IGNORECASE)
        if id_match:
            spreadsheet_id = id_match.group(1)
            
            # Batch update multiple cells
            batch_result = await mcp_session.call_tool(
                'external_google-sheets_batch_update',
                {
                    'spreadsheet_id': spreadsheet_id,
                    'updates': [
                        {'range': 'A1', 'value': 'New A1'},
                        {'range': 'B2', 'value': 'New B2'},
                        {'range': 'C1:C2', 'values': [['New C1'], ['New C2']]}
                    ]
                }
            )
            
            assert batch_result is not None
            print(f"Batch update result: {batch_result}")
    
    @pytest.mark.asyncio
    async def test_google_sheets_error_handling(self, mcp_session):
        """Test error handling for invalid operations"""
        # Try to read non-existent spreadsheet
        with pytest.raises(Exception) as exc_info:
            await mcp_session.call_tool(
                'external_google-sheets_read_spreadsheet',
                {
                    'spreadsheet_id': 'invalid_id_12345',
                    'range': 'A1'
                }
            )
        
        # The error should mention something about the spreadsheet not being found
        error_msg = str(exc_info.value).lower()
        assert 'not found' in error_msg or 'invalid' in error_msg or 'error' in error_msg