"""
FastMCP tools for managing MCP clients and their tools
"""

from typing import Dict, Any, List, Optional
from logai_mcp.session import app
from logai_mcp.mcp_manager import get_mcp_manager, MCPClientConfig


@app.tool()
async def mcp_list_clients() -> Dict[str, Any]:
    """
    List all configured MCP clients and their connection status
    
    Returns information about all MCP clients including their status,
    configuration, and available tools.
    """
    try:
        manager = get_mcp_manager()
        clients = manager.list_clients()
        
        # Add tool counts
        all_tools = await manager.get_all_tools()
        for client_name, client_info in clients.items():
            client_info["tool_count"] = len(all_tools.get(client_name, []))
            client_info["tools"] = [t["name"] for t in all_tools.get(client_name, [])]
        
        return {
            "success": True,
            "clients": clients,
            "total_clients": len(clients),
            "connected_clients": sum(1 for c in clients.values() if c["connected"])
        }
    except Exception as e:
        return {"error": f"Error listing MCP clients: {str(e)}"}


@app.tool()
async def mcp_list_tools() -> Dict[str, Any]:
    """
    List all available tools from all connected MCP clients
    
    Returns a comprehensive list of all tools available across all
    connected MCP clients, including their parameters and descriptions.
    """
    try:
        manager = get_mcp_manager()
        all_tools = await manager.get_all_tools()
        
        # Flatten for easier consumption
        flattened_tools = []
        for client_name, tools in all_tools.items():
            for tool in tools:
                flattened_tools.append({
                    "client": client_name,
                    "name": tool["name"],
                    "fastmcp_name": tool["fastmcp_name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                })
        
        return {
            "success": True,
            "tools_by_client": all_tools,
            "all_tools": flattened_tools,
            "total_tools": len(flattened_tools)
        }
    except Exception as e:
        return {"error": f"Error listing MCP tools: {str(e)}"}


@app.tool()
async def mcp_add_client(
    name: str,
    command: str,
    args: List[str],
    env: Optional[Dict[str, str]] = None,
    description: str = "",
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Add a new MCP client configuration and connect to it
    
    Args:
        name: Unique name for the MCP client
        command: Command to run the MCP server (e.g., 'python', 'node', 'npx')
        args: Arguments to pass to the command
        env: Environment variables to set for the server
        description: Human-readable description of what this client does
        timeout: Connection timeout in seconds
    
    Returns:
        Success status and connection details
    """
    try:
        manager = get_mcp_manager()
        
        config = MCPClientConfig(
            name=name,
            command=command,
            args=args,
            env=env or {},
            description=description,
            timeout=timeout,
            auto_retry=True
        )
        
        success = await manager.add_client(config)
        
        if success:
            # Get the tools that were registered
            all_tools = await manager.get_all_tools()
            client_tools = all_tools.get(name, [])
            
            return {
                "success": True,
                "message": f"Successfully added and connected to MCP client '{name}'",
                "client": name,
                "tools_registered": len(client_tools),
                "tools": [t["fastmcp_name"] for t in client_tools]
            }
        else:
            return {
                "success": False,
                "error": f"Failed to connect to MCP client '{name}'. Check logs for details."
            }
            
    except Exception as e:
        return {"error": f"Error adding MCP client: {str(e)}"}


@app.tool()
async def mcp_remove_client(name: str) -> Dict[str, Any]:
    """
    Remove an MCP client and close its connection
    
    Args:
        name: Name of the MCP client to remove
    
    Returns:
        Success status and removal details
    """
    try:
        manager = get_mcp_manager()
        success = await manager.remove_client(name)
        
        if success:
            return {
                "success": True,
                "message": f"Successfully removed MCP client '{name}'"
            }
        else:
            return {
                "success": False,
                "error": f"Failed to remove MCP client '{name}'"
            }
            
    except Exception as e:
        return {"error": f"Error removing MCP client: {str(e)}"}


@app.tool()
async def mcp_reload_clients() -> Dict[str, Any]:
    """
    Reload all MCP client connections from configuration
    
    This will disconnect all current clients and reconnect them based
    on the current configuration file. Useful after making manual
    configuration changes.
    
    Returns:
        Success status and reconnection details
    """
    try:
        manager = get_mcp_manager()
        await manager.reload_all_clients()
        
        clients = manager.list_clients()
        connected_count = sum(1 for c in clients.values() if c["connected"])
        
        return {
            "success": True,
            "message": "Successfully reloaded all MCP clients",
            "total_clients": len(clients),
            "connected_clients": connected_count,
            "clients": list(clients.keys())
        }
        
    except Exception as e:
        return {"error": f"Error reloading MCP clients: {str(e)}"}


@app.tool()
async def mcp_call_tool(
    client_name: str,
    tool_name: str,
    arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Directly call a tool from a specific MCP client
    
    This is a fallback method for calling MCP tools when the auto-generated
    FastMCP tools are not working as expected.
    
    Args:
        client_name: Name of the MCP client
        tool_name: Name of the tool to call
        arguments: Arguments to pass to the tool
    
    Returns:
        Tool execution results
    """
    try:
        manager = get_mcp_manager()
        session = manager.get_session(client_name)
        
        if not session:
            return {
                "error": f"MCP client '{client_name}' is not connected",
                "available_clients": list(manager.active_sessions.keys())
            }
        
        result = await session.call_tool(tool_name, arguments)
        
        return {
            "success": True,
            "client": client_name,
            "tool": tool_name,
            "result": result.content
        }
        
    except Exception as e:
        return {
            "error": f"Error calling {client_name}.{tool_name}: {str(e)}",
            "client": client_name,
            "tool": tool_name
        }


@app.tool()
async def mcp_client_status(name: str) -> Dict[str, Any]:
    """
    Get detailed status information for a specific MCP client
    
    Args:
        name: Name of the MCP client to check
    
    Returns:
        Detailed status and configuration information
    """
    try:
        manager = get_mcp_manager()
        clients = manager.list_clients()
        
        if name not in clients:
            return {
                "error": f"MCP client '{name}' not found",
                "available_clients": list(clients.keys())
            }
        
        client_info = clients[name]
        
        # Get tools for this client
        all_tools = await manager.get_all_tools()
        client_tools = all_tools.get(name, [])
        
        return {
            "success": True,
            "client": name,
            "status": client_info,
            "tools": client_tools,
            "tool_count": len(client_tools)
        }
        
    except Exception as e:
        return {"error": f"Error getting client status: {str(e)}"}


@app.tool()
async def mcp_test_connection(name: str) -> Dict[str, Any]:
    """
    Test the connection to a specific MCP client
    
    Args:
        name: Name of the MCP client to test
    
    Returns:
        Connection test results
    """
    try:
        manager = get_mcp_manager()
        session = manager.get_session(name)
        
        if not session:
            return {
                "success": False,
                "client": name,
                "error": "Client not connected"
            }
        
        # Try to list tools as a connection test
        tools_result = await session.list_tools()
        
        return {
            "success": True,
            "client": name,
            "message": "Connection test successful",
            "tool_count": len(tools_result.tools),
            "tools": [tool.name for tool in tools_result.tools]
        }
        
    except Exception as e:
        return {
            "success": False,
            "client": name,
            "error": f"Connection test failed: {str(e)}"
        } 