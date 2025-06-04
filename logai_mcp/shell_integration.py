"""
IPython shell integration for MCP tools

This module handles creating convenience functions in the IPython shell
that mirror the dynamically registered FastMCP tools.
"""

import logging
from typing import Dict, Any, Callable, Optional
from logai_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from logai_mcp.mcp_manager import get_mcp_manager

logger = logging.getLogger(__name__)


async def update_shell_functions():
    """
    Generate and push convenience functions to IPython shell
    
    This creates shell functions that mirror the auto-generated FastMCP tools,
    allowing users to call MCP tools directly from the shell with proper
    function signatures and documentation.
    """
    try:
        manager = get_mcp_manager()
        
        # Get all available tools from active MCP clients
        all_tools = await manager.get_all_tools()
        
        # Generate convenience functions for each tool
        functions_to_push = {}
        function_definitions = []
        
        for client_name, tools in all_tools.items():
            for tool in tools:
                # Create namespaced function name (same as FastMCP tool)
                func_name = f"{client_name}_{tool['name']}"
                
                # Generate the function code
                func_code = _generate_shell_function(client_name, tool)
                function_definitions.append(func_code)
                
                # Create the actual function
                shell_function = _make_shell_function(client_name, tool)
                functions_to_push[func_name] = shell_function
        
        # Execute all function definitions in the shell
        if function_definitions:
            combined_code = "\n\n".join(function_definitions)
            await run_code_in_shell(combined_code)
        
        # Push utility functions
        utility_functions = {
            "mcp_tools": all_tools,
            "call_mcp": _create_direct_mcp_caller(),
            "list_mcp_tools": _create_tool_lister(),
            "mcp_status": _create_status_checker()
        }
        
        functions_to_push.update(utility_functions)
        
        # Push all functions to IPython shell
        _SHELL.push(functions_to_push)
        
        total_tools = sum(len(tools) for tools in all_tools.values())
        logger.info(f"Updated shell with {total_tools} MCP tool functions from {len(all_tools)} clients")
        
    except Exception as e:
        logger.error(f"Error updating shell functions: {e}")


def _generate_shell_function(client_name: str, tool: Dict[str, Any]) -> str:
    """Generate Python code for a shell function"""
    func_name = f"{client_name}_{tool['name']}"
    
    # Extract parameters from tool schema
    params = []
    docstring_params = []
    
    if 'parameters' in tool and 'properties' in tool['parameters']:
        properties = tool['parameters']['properties']
        required = tool['parameters'].get('required', [])
        
        for param_name, param_def in properties.items():
            param_type = param_def.get('type', 'str')
            param_desc = param_def.get('description', '')
            is_required = param_name in required
            
            # Convert JSON schema types to Python
            type_hint = {
                'string': 'str',
                'integer': 'int', 
                'number': 'float',
                'boolean': 'bool',
                'array': 'list',
                'object': 'dict'
            }.get(param_type, 'str')
            
            if is_required:
                params.append(f"{param_name}: {type_hint}")
            else:
                params.append(f"{param_name}: {type_hint} = None")
            
            docstring_params.append(f"        {param_name}: {param_desc}")
    
    param_string = ", ".join(params)
    docstring_param_section = "\n".join(docstring_params) if docstring_params else "        No parameters"
    
    # Generate the function code
    function_code = f'''
async def {func_name}({param_string}):
    """
    [MCP:{client_name}] {tool['description']}
    
    Parameters:
{docstring_param_section}
    
    Returns:
        Tool execution result
    """
    import asyncio
    from logai_mcp.mcp_manager import get_mcp_manager
    
    try:
        manager = get_mcp_manager()
        session = manager.get_session("{client_name}")
        
        if not session:
            return {{"error": "MCP client '{client_name}' is not connected"}}
        
        # Build arguments dict, excluding None values
        kwargs = {{}}
        {_generate_kwargs_code(tool)}
        
        result = await session.call_tool("{tool['name']}", kwargs)
        return result.content
        
    except Exception as e:
        return {{"error": f"Error calling {client_name}.{tool['name']}: {{str(e)}}"}}
'''
    
    return function_code


def _generate_kwargs_code(tool: Dict[str, Any]) -> str:
    """Generate code for building kwargs dict"""
    if 'parameters' not in tool or 'properties' not in tool['parameters']:
        return ""
    
    lines = []
    for param_name in tool['parameters']['properties'].keys():
        lines.append(f'        if {param_name} is not None:')
        lines.append(f'            kwargs["{param_name}"] = {param_name}')
    
    return "\n".join(lines)


def _make_shell_function(client_name: str, tool: Dict[str, Any]) -> Callable:
    """Create a callable function for the shell"""
    
    async def shell_function(**kwargs):
        f"""[MCP:{client_name}] {tool['description']}"""
        try:
            manager = get_mcp_manager()
            session = manager.get_session(client_name)
            
            if not session:
                return {"error": f"MCP client '{client_name}' is not connected"}
            
            # Filter out None values
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            
            result = await session.call_tool(tool['name'], filtered_kwargs)
            return result.content
            
        except Exception as e:
            return {"error": f"Error calling {client_name}.{tool['name']}: {str(e)}"}
    
    # Set proper metadata
    shell_function.__name__ = f"{client_name}_{tool['name']}"
    shell_function.__doc__ = f"[MCP:{client_name}] {tool['description']}"
    
    return shell_function


def _create_direct_mcp_caller() -> Callable:
    """Create a direct MCP tool caller utility function"""
    
    async def call_mcp(client_name: str, tool_name: str, **kwargs):
        """
        Direct caller for MCP tools
        
        Args:
            client_name: Name of the MCP client
            tool_name: Name of the tool to call
            **kwargs: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            manager = get_mcp_manager()
            session = manager.get_session(client_name)
            
            if not session:
                return {"error": f"MCP client '{client_name}' is not connected"}
            
            result = await session.call_tool(tool_name, kwargs)
            return result.content
            
        except Exception as e:
            return {"error": f"Error calling {client_name}.{tool_name}: {str(e)}"}
    
    return call_mcp


def _create_tool_lister() -> Callable:
    """Create a tool listing utility function"""
    
    async def list_mcp_tools(client_name: Optional[str] = None):
        """
        List available MCP tools
        
        Args:
            client_name: Optional client name to filter by
            
        Returns:
            Dictionary of available tools
        """
        try:
            manager = get_mcp_manager()
            all_tools = await manager.get_all_tools()
            
            if client_name:
                return all_tools.get(client_name, [])
            
            return all_tools
            
        except Exception as e:
            return {"error": f"Error listing tools: {str(e)}"}
    
    return list_mcp_tools


def _create_status_checker() -> Callable:
    """Create a status checking utility function"""
    
    def mcp_status():
        """
        Get MCP client status information
        
        Returns:
            Dictionary of client statuses
        """
        try:
            manager = get_mcp_manager()
            return manager.list_clients()
            
        except Exception as e:
            return {"error": f"Error getting status: {str(e)}"}
    
    return mcp_status


async def refresh_mcp_shell_functions():
    """
    Refresh all MCP shell functions
    
    This is a convenience function that can be called from the shell
    to refresh all MCP tool functions after configuration changes.
    """
    await update_shell_functions()
    return "MCP shell functions refreshed successfully"


# Push the refresh function to the shell immediately
_SHELL.push({"refresh_mcp_shell_functions": refresh_mcp_shell_functions}) 