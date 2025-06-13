# Integration Tests

This directory contains integration tests for the LogAI MCP server.

## Test Files

### SDK-Based Tests (Recommended)
These tests use the official MCP Python SDK client with streamable HTTP transport:
- `test_basic_connection_sdk.py` - Basic connection and health check tests
- `test_google_sheets_sdk.py` - Google Sheets integration tests
- `test_multi_mcp_integration_sdk.py` - Multi-MCP server integration tests

### Legacy HTTP Tests (Deprecated)
These tests use direct HTTP POST requests and are kept for backward compatibility:
- `test_basic_connection.py` (deprecated - use `test_basic_connection_sdk.py`)
- `test_google_sheets.py` (deprecated - use `test_google_sheets_sdk.py`) 
- `test_multi_mcp_integration.py` (deprecated - use `test_multi_mcp_integration_sdk.py`)

## Running Tests

### Using Docker (Recommended)
```bash
# Run integration tests in Docker environment
./run-integration-tests.sh
```

### Running Locally
```bash
# Run all integration tests
pytest tests/integration/

# Run only SDK-based tests
pytest tests/integration/test_*_sdk.py

# Run a specific test file
pytest tests/integration/test_basic_connection_sdk.py
```

## Environment Variables

- `MCP_SERVER_URL` - Base URL of the MCP server (default: `http://localhost:8000`)

## Using the MCP SDK

The SDK-based tests use the official MCP Python SDK with streamable HTTP transport:

```python
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

async def test_example():
    base_url = "http://localhost:8000"
    
    # Connect to streamable HTTP server
    async with streamablehttp_client(f"{base_url}/mcp/") as (read_stream, write_stream, _):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            
            # Call a tool
            result = await session.call_tool("tool_name", {"param": "value"})
```

## Key Advantages

1. **Official SDK**: Uses the official MCP Python SDK
2. **Protocol Compliance**: Ensures compatibility with MCP protocol
3. **Type Safety**: Proper type definitions for all operations
4. **Cleaner Code**: No need to manually handle HTTP requests or JSON-RPC
5. **Better Error Handling**: SDK handles protocol-level errors properly