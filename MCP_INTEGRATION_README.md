# MCP Integration System

This system allows you to add arbitrary Model Context Protocol (MCP) servers to your logai-mcp application, similar to how Claude Desktop manages MCP servers. The integration creates dynamic FastMCP tools and IPython shell functions for each MCP tool.

## How It Works

1. **Configuration**: MCP servers are defined in `mcp_config.json`
2. **Dynamic Connection**: The MCP manager connects to each server using the Python MCP SDK
3. **Tool Registration**: Each MCP tool becomes a FastMCP tool with the pattern `{client_name}_{tool_name}`
4. **Shell Integration**: Convenience functions are created in the IPython shell
5. **Runtime Management**: Add/remove MCP servers without restarting

## Configuration

Edit `mcp_config.json` to add MCP servers:

```json
{
  "mcp_clients": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "env": {},
      "description": "Secure file operations",
      "timeout": 30,
      "auto_retry": true
    },
    "fetch": {
      "command": "npx", 
      "args": ["-y", "@smithery-ai/fetch"],
      "env": {},
      "description": "Web content fetching",
      "timeout": 30,
      "auto_retry": true
    }
  }
}
```

## Pre-configured Popular MCPs

The system comes with these popular MCPs pre-configured:

1. **Sequential Thinking** - Most popular MCP (5,550+ uses)
   - Structured problem-solving capabilities
   - No authentication required

2. **Filesystem** - Official reference server
   - Secure file operations with access controls
   - Uses `/tmp` directory for safety

3. **Fetch** - Simple web tool
   - Web content fetching and conversion
   - No authentication required

4. **Memory** - Knowledge graph memory
   - Persistent memory across sessions
   - Local knowledge graph storage

## Usage

### Via MCP Tools (for LLMs)

When the server is running, LLMs can use these tools:

- `mcp_list_clients()` - List all configured MCP clients
- `mcp_list_tools()` - List all available tools from MCP clients  
- `mcp_add_client(name, command, args, ...)` - Add new MCP client
- `mcp_remove_client(name)` - Remove MCP client
- `mcp_reload_clients()` - Reload all clients from config
- `{client_name}_{tool_name}(...)` - Direct tool calls (auto-generated)

### Via IPython Shell

```python
# List MCP status
mcp_status()

# List all tools
await list_mcp_tools()

# Call MCP tools directly
await filesystem_list_directory(path="/tmp")
await fetch_fetch(url="https://example.com")

# Direct MCP call
await call_mcp("filesystem", "list_directory", path="/tmp")

# Refresh shell functions after config changes
await refresh_mcp_shell_functions()
```

## Testing

Run the test script to verify everything works:

```bash
python test_mcp_integration.py
```

This will:
1. Initialize the MCP manager
2. Connect to configured MCP servers
3. List available tools
4. Update shell functions
5. Verify tool call interfaces

## Architecture

### Key Components

1. **MCPManager** (`logai_mcp/mcp_manager.py`)
   - Manages MCP client connections
   - Handles configuration loading/saving
   - Provides connection pooling and retry logic

2. **DynamicToolRegistry** 
   - Auto-registers MCP tools as FastMCP tools
   - Creates proper function signatures
   - Handles tool name conflicts with prefixing

3. **Shell Integration** (`logai_mcp/shell_integration.py`)
   - Creates convenience functions for IPython
   - Generates proper function signatures and docstrings
   - Provides utility functions for tool management

4. **MCP Tools** (`logai_mcp/mcp_tools.py`)
   - FastMCP tools for managing MCP clients
   - Available to LLMs via the Model Context Protocol

### Tool Naming Convention

- **FastMCP Tools**: `{client_name}_{tool_name}` (e.g., `filesystem_list_directory`)
- **Shell Functions**: Same as FastMCP tools but available as Python functions
- **Unique IDs**: Prevents naming conflicts between different MCP servers

### Connection Management

- **Graceful Failures**: Failed connections don't crash the system
- **Auto-retry**: Configurable retry logic for failed connections
- **Timeout Handling**: Configurable timeouts prevent hanging
- **Hot Reloading**: Add/remove servers without restart

## Error Handling

- Failed connections are logged but don't prevent other servers from connecting
- Individual tool failures return error objects instead of crashing
- Configuration errors are validated before attempting connections
- Network timeouts are handled gracefully

## Running Server Refresh

The system supports adding new MCP servers while the FastMCP server is running:

1. **Add to config**: Update `mcp_config.json`
2. **Reload via tool**: Use `mcp_reload_clients()` tool
3. **Or restart**: Restart the FastMCP server

The shell functions are automatically refreshed when MCP clients are added/removed.

## Popular MCP Servers to Try

Here are some popular MCP servers you can easily add:

```bash
# GitHub (requires GITHUB_TOKEN)
npx -y @smithery-ai/github

# Git operations
uvx mcp-server-git --repository /path/to/repo

# Brave Search (requires API key)
npx -y @smithery-ai/brave-search

# SQLite
npx -y @modelcontextprotocol/server-sqlite /path/to/db.sqlite

# Postgres (requires connection string)
npx -y @modelcontextprotocol/server-postgres postgresql://...
```

## Security Considerations

- File system access is restricted to specified directories
- MCP servers run in separate processes
- Environment variables can be used for API keys
- Timeout protection prevents resource exhaustion
- No arbitrary code execution in the main process

## Troubleshooting

### Common Issues

1. **NPX not found**: Install Node.js and npm
2. **UVX not found**: Install `uv` Python tool
3. **Connection timeouts**: Increase timeout in config
4. **Tool not found**: Check if MCP server is properly connected

### Debug Mode

Set logging level to DEBUG for detailed information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Status

```python
# In IPython shell
mcp_status()  # Shows connection status
await list_mcp_tools()  # Shows available tools
```

This integration system makes it easy to extend your LogAI MCP server with any MCP-compatible tools while maintaining security and reliability. 