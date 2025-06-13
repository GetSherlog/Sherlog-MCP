"""External MCP Tools for LogAI MCP Server

This module provides dynamic integration with external MCP servers.
It discovers and registers tools from configured MCP servers at runtime,
making them available as native LogAI tools with DataFrame integration.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from logai_mcp.session import app, logger
from logai_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from logai_mcp.config import get_settings


# Store registered external tools info for documentation
EXTERNAL_TOOLS_REGISTRY: Dict[str, Dict[str, types.Tool]] = {}


async def auto_register_external_mcps():
    """Automatically discover and register all tools from configured external MCPs.

    This function is called during server startup to dynamically add tools
    from external MCP servers to the LogAI MCP server.
    """
    settings = get_settings()
    external_mcps = settings.external_mcps

    if not external_mcps:
        logger.info("No external MCPs configured")
        return

    logger.info(f"Registering external MCPs: {list(external_mcps.keys())}")

    for mcp_name, mcp_config in external_mcps.items():
        try:
            await register_mcp_tools(mcp_name, mcp_config)
        except Exception as e:
            logger.error(f"Failed to register MCP '{mcp_name}': {e}")


async def register_mcp_tools(mcp_name: str, mcp_config: Dict[str, Any]):
    """Register all tools from a specific external MCP server.
    
    Args:
        mcp_name: Name identifier for the MCP server
        mcp_config: Configuration dict with command, args, and env
    """
    logger.info(f"Connecting to external MCP: {mcp_name}")
    
    # Validate configuration
    if not mcp_config.get("command"):
        raise ValueError(f"Missing 'command' in configuration for {mcp_name}")
    
    # Prepare environment variables
    env = os.environ.copy()
    if "env" in mcp_config:
        for key, value in mcp_config["env"].items():
            if isinstance(value, str):
                # Support environment variable expansion for backward compatibility
                if value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    env_value = os.environ.get(env_var, "")
                    if not env_value and key in [
                    "SERVICE_ACCOUNT_PATH", "DATABASE_URL", "API_KEY", "GITHUB_PERSONAL_ACCESS_TOKEN"
                ]:
                        logger.warning(f"Environment variable {env_var} not set for {mcp_name}")
                    env[key] = env_value
                else:
                    # Direct value from mcp.json
                    env[key] = value
            else:
                env[key] = str(value)
    
    # Create stdio parameters
    stdio_params = StdioServerParameters(
        command=mcp_config["command"],
        args=mcp_config.get("args", []),
        env=env
    )
    
    try:
        # Use stdio_client for connection
        async with stdio_client(stdio_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection with timeout
                try:
                    await asyncio.wait_for(session.initialize(), timeout=30.0)
                except asyncio.TimeoutError:
                    raise Exception(f"Timeout connecting to {mcp_name} MCP server")
                
                # List available tools
                try:
                    tools_response = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                    tools = tools_response.tools if hasattr(tools_response, 'tools') else []
                except asyncio.TimeoutError:
                    raise Exception(f"Timeout listing tools from {mcp_name}")
                
                logger.info(f"Found {len(tools)} tools in {mcp_name}")
                
                # Store tools info for registry
                EXTERNAL_TOOLS_REGISTRY[mcp_name] = {}
                
                # Register each tool
                registered_count = 0
                for tool in tools:
                    try:
                        register_external_tool(mcp_name, tool, mcp_config)
                        EXTERNAL_TOOLS_REGISTRY[mcp_name][tool.name] = tool
                        registered_count += 1
                    except Exception as e:
                        logger.error(f"Failed to register tool {tool.name} from {mcp_name}: {e}")
                
                logger.info(f"Successfully registered {registered_count}/{len(tools)} tools from {mcp_name}")
        
    except Exception as e:
        logger.error(f"Error connecting to {mcp_name} MCP: {e}")
        raise


def register_external_tool(mcp_name: str, tool_info: types.Tool, mcp_config: Dict[str, Any]):
    """Register a single external MCP tool with the LogAI server.
    
    Args:
        mcp_name: Name of the MCP server
        tool_info: Tool information from the external MCP
        mcp_config: Configuration for connecting to the MCP
    """
    # Generate unique tool name
    full_tool_name = f"external_{mcp_name}_{tool_info.name}"
    
    logger.debug(f"Registering tool: {full_tool_name}")
    
    # Build comprehensive documentation
    doc_lines = []
    if tool_info.description:
        doc_lines.append(tool_info.description)
    doc_lines.append(f"\n[External MCP: {mcp_name}]")
    doc_lines.append(f"[Original tool: {tool_info.name}]")
    
    # Add parameter documentation if available
    if hasattr(tool_info, 'inputSchema') and tool_info.inputSchema:
        doc_lines.append("\nParameters:")
        schema = tool_info.inputSchema
        if isinstance(schema, dict) and 'properties' in schema:
            for param_name, param_info in schema['properties'].items():
                param_type = param_info.get('type', 'any')
                param_desc = param_info.get('description', '')
                required = param_name in schema.get('required', [])
                req_str = " (required)" if required else " (optional)"
                doc_lines.append(f"  {param_name}: {param_type}{req_str} - {param_desc}")
    
    doc_lines.append("\n  save_as: str - Variable name to store results in IPython shell")
    
    # Create a factory function that returns the actual tool function
    def create_tool_function():
        # This function will be called for each tool invocation
        async def tool_impl(**kwargs) -> Any:
            # Extract our special parameter
            save_as = kwargs.pop('save_as', f"{full_tool_name}_result")
            
            # Prepare the code to run in IPython shell
            code = generate_tool_execution_code(
                mcp_name=mcp_name,
                mcp_config=mcp_config,
                tool_name=tool_info.name,
                params=kwargs,
                save_as=save_as
            )
            
            # Execute in shell
            result = await run_code_in_shell(code)
            
            # Try to return the DataFrame representation if available
            try:
                # Get the actual result from the shell
                shell_result = _SHELL.user_ns.get(save_as)
                if isinstance(shell_result, pd.DataFrame):
                    return shell_result.to_dict('records')
                return result
            except:
                return result
        
        # Set metadata
        tool_impl.__name__ = full_tool_name
        tool_impl.__doc__ = "\n".join(doc_lines)
        
        return tool_impl
    
    # Create the tool function
    tool_func = create_tool_function()
    
    # Use the decorator to register it
    # This is a workaround to dynamically use the decorator
    decorated_func = app.tool()(tool_func)
    
    # Store reference for documentation
    setattr(app, f"_external_tool_{full_tool_name}", decorated_func)


def generate_tool_execution_code(
    mcp_name: str,
    mcp_config: Dict[str, Any],
    tool_name: str,
    params: Dict[str, Any],
    save_as: str
) -> str:
    """Generate Python code to execute an external MCP tool in the IPython shell.
    
    Args:
        mcp_name: Name of the MCP server
        mcp_config: Configuration for the MCP
        tool_name: Name of the tool to call
        params: Parameters to pass to the tool
        save_as: Variable name to save results
        
    Returns:
        Python code string to execute
    """
    # Prepare environment setup
    env_setup = "import os\nenv = os.environ.copy()\n"
    if "env" in mcp_config:
        for key, value in mcp_config["env"].items():
            if isinstance(value, str):
                # Support environment variable expansion
                if value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    env_setup += f"env['{key}'] = os.environ.get('{env_var}', '')\n"
                else:
                    # Direct value from mcp.json
                    env_setup += f"env['{key}'] = {repr(value)}\n"
            else:
                env_setup += f"env['{key}'] = {repr(str(value))}\n"
    
    code = f"""
# Execute external MCP tool: {mcp_name}.{tool_name}
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pandas as pd
import numpy as np

{env_setup}

# Create connection parameters
stdio_params = StdioServerParameters(
    command={repr(mcp_config['command'])},
    args={repr(mcp_config.get('args', []))},
    env=env
)

# Connect and execute tool
async def _execute_external_tool():
    try:
        async with stdio_client(stdio_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize connection with timeout
                try:
                    await asyncio.wait_for(session.initialize(), timeout=30.0)
                except asyncio.TimeoutError:
                    raise Exception(f"Timeout connecting to {mcp_name} MCP server")
                
                # Call the tool with timeout
                try:
                    result = await asyncio.wait_for(
                        session.call_tool({repr(tool_name)}, arguments={repr(params)}),
                        timeout=60.0
                    )
                except asyncio.TimeoutError:
                    raise Exception(f"Timeout calling tool {tool_name} on {mcp_name}")
                
                # Extract the actual result
                if hasattr(result, 'content'):
                    # Handle different content types
                    if isinstance(result.content, list) and len(result.content) > 0:
                        content = result.content[0]
                        if hasattr(content, 'text'):
                            # Try to parse as JSON
                            try:
                                import json
                                return json.loads(content.text)
                            except:
                                return content.text
                        else:
                            return content
                    else:
                        return result.content
                else:
                    return result
                
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error executing {mcp_name}.{tool_name}: {{e}}")
        raise

# Execute the async function
try:
    _result = await _execute_external_tool()
except Exception as e:
    _result = {{"error": str(e), "tool": "{tool_name}", "mcp": "{mcp_name}"}}

# Convert to DataFrame if possible
{save_as} = convert_to_dataframe(_result)

# Display result info
print(f"Result stored in '{save_as}'")
if isinstance({save_as}, pd.DataFrame):
    print(f"DataFrame shape: {{{save_as}.shape}}")
    print(f"Columns: {{{save_as}.columns.tolist()}}")

{save_as}
"""
    
    return code


def convert_to_dataframe(data: Any) -> Union[pd.DataFrame, Any]:
    """Convert various data formats to pandas DataFrame when possible.
    
    Args:
        data: Input data in various formats
        
    Returns:
        DataFrame if conversion successful, original data otherwise
    """
    if data is None:
        return pd.DataFrame()
    
    # Already a DataFrame
    if isinstance(data, pd.DataFrame):
        return data
    
    # List of dictionaries (most common case)
    if isinstance(data, list) and len(data) > 0:
        if all(isinstance(item, dict) for item in data):
            return pd.DataFrame(data)
        # List of lists with potential header
        elif all(isinstance(item, list) for item in data):
            # Assume first row might be headers
            if len(data) > 1:
                try:
                    # Try with first row as header
                    df = pd.DataFrame(data[1:], columns=data[0])
                    return df
                except:
                    # Fall back to no header
                    return pd.DataFrame(data)
            else:
                return pd.DataFrame(data)
    
    # Dictionary with list values (column-oriented data)
    if isinstance(data, dict):
        # Check if all values are lists of same length
        if all(isinstance(v, list) for v in data.values()):
            lengths = [len(v) for v in data.values()]
            if len(set(lengths)) == 1:  # All same length
                return pd.DataFrame(data)
        
        # Single row dictionary
        if all(not isinstance(v, (list, dict)) for v in data.values()):
            return pd.DataFrame([data])
        
        # Nested structure - try to normalize
        try:
            return pd.json_normalize(data)
        except:
            pass
    
    # NumPy array
    if isinstance(data, np.ndarray):
        return pd.DataFrame(data)
    
    # String that might be CSV or JSON
    if isinstance(data, str):
        # Try JSON first
        try:
            import json
            json_data = json.loads(data)
            return convert_to_dataframe(json_data)  # Recursive call
        except:
            pass
        
        # Try CSV
        try:
            from io import StringIO
            return pd.read_csv(StringIO(data))
        except:
            pass
    
    # If all else fails, return original data
    return data


# Push conversion function to IPython shell
_SHELL.push({
    "convert_to_dataframe": convert_to_dataframe
})


@app.tool()
async def list_external_tools(server: Optional[str] = None) -> Dict[str, Any]:
    """List all registered external MCP tools.
    
    Args:
        server: Optional specific server name to filter by
        
    Returns:
        Dictionary with tool information organized by server
    """
    result = {}
    
    for mcp_name, tools in EXTERNAL_TOOLS_REGISTRY.items():
        if server and server != mcp_name:
            continue
            
        server_info = {
            "tools": []
        }
        
        for tool_name, tool_info in tools.items():
            tool_data = {
                "name": tool_name,
                "full_name": f"{mcp_name}_{tool_name}",
                "description": tool_info.description
            }
            
            # Add parameter info if available
            if hasattr(tool_info, 'inputSchema') and tool_info.inputSchema:
                schema = tool_info.inputSchema
                if isinstance(schema, dict) and 'properties' in schema:
                    tool_data["parameters"] = {
                        "properties": schema['properties'],
                        "required": schema.get('required', [])
                    }
            
            server_info["tools"].append(tool_data)
        
        result[mcp_name] = server_info
    
    return result