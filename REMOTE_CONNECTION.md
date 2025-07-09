# Connecting to Sherlog MCP as a Remote Server

This guide explains how to connect to Sherlog MCP when it's running as a remote server.

## Deployment

### 1. Build and Run with Docker

```bash
# Build the Docker image
docker build -t sherlog-mcp .

# Run the server
docker run -p 8000:8000 sherlog-mcp
```

The server will be available at `http://localhost:8000/mcp`

## Connecting from Claude Desktop

Claude Desktop doesn't natively support remote MCP servers yet, but you can use the `mcp-remote` proxy:

### 1. Install mcp-remote

Make sure you have Node.js installed, then you can use `npx` directly.

### 2. Configure Claude Desktop

Edit your Claude Desktop configuration (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "sherlog": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://your-server-ip:8000/mcp"
      ]
    }
  }
}
```

Replace `your-server-ip` with:
- `localhost` if running locally
- Your server's IP address or domain if running remotely

### 3. Restart Claude Desktop

After saving the configuration, restart Claude Desktop. The Sherlog tools should appear in the tools menu.

## Direct HTTP Client Connection

For other MCP clients that support streamable-http transport, you can connect directly:

- **URL**: `http://your-server:8000/mcp`
- **Transport**: `streamable-http`
- **Authentication**: None required (add your own if needed)

## Available Tools

Once connected, you'll have access to:

- **call_cli()** - Execute CLI commands
- **search_pypi()** - Search for Python packages
- **execute_python_code()** - Run Python code
- **Code retrieval tools** - Search and analyze codebases

## Setting Up CLIs in the Container

To use CLI tools like GitHub CLI:

1. Access the running container:
   ```bash
   docker exec -it <container-id> /bin/bash
   ```

2. Install and authenticate CLIs:
   ```bash
   # Install GitHub CLI
   apt-get update && apt-get install -y gh
   
   # Authenticate
   gh auth login
   ```

3. Exit the container. The CLI is now available for use through the MCP tools.

## Troubleshooting

### Connection Issues

1. Ensure the server is running and accessible
2. Check firewall rules allow port 8000
3. Verify the URL in your configuration is correct

### mcp-remote Issues

If you encounter issues with mcp-remote:

```bash
# Install globally instead of using npx
npm install -g mcp-remote

# Then update config to use global install
"command": "mcp-remote"
```

### Debugging

Check server logs:
```bash
docker logs <container-id>
```

## Security Considerations

- The server listens on all interfaces (0.0.0.0) by default
- No authentication is built-in - add your own if needed
- Use HTTPS in production with a reverse proxy
- Consider network security (VPN, firewall rules)