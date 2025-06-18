"""External MCP Tools for LogAI MCP Server

This module provides dynamic integration with external MCP servers.
It discovers and registers tools from configured MCP servers at runtime,
making them available as native LogAI tools with DataFrame integration.
"""

import asyncio
import inspect
import os
from typing import Any, Optional

import pandas as pd
import polars as pl
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from logai_mcp.config import get_settings
from logai_mcp.dataframe_utils import smart_create_dataframe, to_json_serializable
from logai_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from logai_mcp.session import app, logger


def _external_mcps_available() -> bool:
    """Check if external MCPs are configured."""
    try:
        settings = get_settings()
        return bool(settings.external_mcps_json or settings.external_mcps)
    except Exception:
        return False


EXTERNAL_TOOLS_REGISTRY: dict[str, dict[str, types.Tool]] = {}


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


async def register_mcp_tools(mcp_name: str, mcp_config: dict[str, Any]):
    """Register all tools from a specific external MCP server.

    Args:
        mcp_name: Name identifier for the MCP server
        mcp_config: Configuration dict with command, args, and env

    """
    logger.info(f"Connecting to external MCP: {mcp_name}")

    if not mcp_config.get("command"):
        raise ValueError(f"Missing 'command' in configuration for {mcp_name}")

    env = os.environ.copy()
    if "env" in mcp_config:
        for key, value in mcp_config["env"].items():
            if isinstance(value, str):
                if value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    env_value = os.environ.get(env_var, "")
                    if not env_value and key in [
                        "SERVICE_ACCOUNT_PATH",
                        "DATABASE_URL",
                        "API_KEY",
                        "GITHUB_PERSONAL_ACCESS_TOKEN",
                    ]:
                        logger.warning(
                            f"Environment variable {env_var} not set for {mcp_name}"
                        )
                    env[key] = env_value
                else:
                    env[key] = value
            else:
                env[key] = str(value)

    stdio_params = StdioServerParameters(
        command=mcp_config["command"], args=mcp_config.get("args", []), env=env
    )

    try:
        async with stdio_client(stdio_params) as (read, write):
            async with ClientSession(read, write) as session:
                try:
                    await asyncio.wait_for(session.initialize(), timeout=30.0)
                except asyncio.TimeoutError:
                    raise Exception(f"Timeout connecting to {mcp_name} MCP server")
                try:
                    tools_response = await asyncio.wait_for(
                        session.list_tools(), timeout=10.0
                    )
                    tools = (
                        tools_response.tools if hasattr(tools_response, "tools") else []
                    )
                except asyncio.TimeoutError:
                    raise Exception(f"Timeout listing tools from {mcp_name}")

                logger.info(f"Found {len(tools)} tools in {mcp_name}")

                EXTERNAL_TOOLS_REGISTRY[mcp_name] = {}

                registered_count = 0
                for tool in tools:
                    try:
                        register_external_tool(mcp_name, tool, mcp_config)
                        EXTERNAL_TOOLS_REGISTRY[mcp_name][tool.name] = tool
                        registered_count += 1
                    except Exception as e:
                        logger.error(
                            f"Failed to register tool {tool.name} from {mcp_name}: {e}"
                        )

                logger.info(
                    f"Successfully registered {registered_count}/{len(tools)} tools from {mcp_name}"
                )

    except Exception as e:
        logger.error(f"Error connecting to {mcp_name} MCP: {e}")
        raise


def _create_parameter_from_schema(param_name: str, param_info: dict) -> inspect.Parameter:
    """Create an inspect.Parameter from JSON schema property info."""
    param_type = param_info.get("type", "string")
    default_value = param_info.get("default", inspect.Parameter.empty)
    
    # Map JSON schema types to Python types for better introspection
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    
    annotation = type_mapping.get(param_type, Any)
    
    # If no default provided but parameter is not required, make it optional
    if default_value == inspect.Parameter.empty:
        annotation = Optional[annotation]
        default_value = None
    
    return inspect.Parameter(
        param_name,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        default=default_value,
        annotation=annotation
    )


def register_external_tool(
    mcp_name: str, tool_info: types.Tool, mcp_config: dict[str, Any]
):
    """Register a single external MCP tool with the LogAI server.

    Args:
        mcp_name: Name of the MCP server
        tool_info: Tool information from the external MCP
        mcp_config: Configuration for connecting to the MCP

    """
    full_tool_name = f"external_{mcp_name}_{tool_info.name}"

    logger.debug(f"Registering tool: {full_tool_name}")

    # Build docstring
    doc_lines = []
    if tool_info.description:
        doc_lines.append(tool_info.description)
    doc_lines.append(f"\n[External MCP: {mcp_name}]")
    doc_lines.append(f"[Original tool: {tool_info.name}]")

    # Extract parameter information from schema
    parameters = []
    required_params = set()
    
    if hasattr(tool_info, "inputSchema") and tool_info.inputSchema:
        doc_lines.append("\nParameters:")
        schema = tool_info.inputSchema
        if isinstance(schema, dict) and "properties" in schema:
            required_params = set(schema.get("required", []))
            
            for param_name, param_info in schema["properties"].items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                required = param_name in required_params
                req_str = " (required)" if required else " (optional)"
                doc_lines.append(
                    f"  {param_name}: {param_type}{req_str} - {param_desc}"
                )
                
                # Create parameter for function signature
                try:
                    param = _create_parameter_from_schema(param_name, param_info)
                    if required and param.default != inspect.Parameter.empty:
                        # Required parameters should not have defaults
                        param = param.replace(default=inspect.Parameter.empty)
                    parameters.append(param)
                except Exception as e:
                    logger.warning(f"Failed to create parameter {param_name}: {e}")

    # Always add save_as parameter
    save_as_param = inspect.Parameter(
        "save_as",
        inspect.Parameter.KEYWORD_ONLY,
        default=f"{full_tool_name}_result",
        annotation=str
    )
    parameters.append(save_as_param)
    
    doc_lines.append(
        "\n  save_as: str - Variable name to store results in IPython shell"
    )

    # Create function signature
    signature = inspect.Signature(parameters)

    def create_tool_function():
        async def tool_impl(*args, **kwargs) -> Any:
            # Bind arguments to signature to validate them
            try:
                bound_args = signature.bind(*args, **kwargs)
                bound_args.apply_defaults()
            except TypeError as e:
                raise ValueError(f"Invalid arguments for {tool_info.name}: {e}")
            
            # Extract save_as and remove from params sent to external tool
            call_params = dict(bound_args.arguments)
            save_as = call_params.pop("save_as", f"{full_tool_name}_result")
            
            # Validate required parameters
            missing_required = []
            for param_name in required_params:
                if param_name not in call_params or call_params[param_name] is None:
                    missing_required.append(param_name)
            
            if missing_required:
                raise ValueError(
                    f"Missing required parameters for {tool_info.name}: {missing_required}"
                )

            code = generate_tool_execution_code(
                mcp_name=mcp_name,
                mcp_config=mcp_config,
                tool_name=tool_info.name,
                params=call_params,
                save_as=save_as,
            )

            result = await run_code_in_shell(code)

            try:
                shell_result = _SHELL.user_ns.get(save_as)
                if isinstance(shell_result, (pd.DataFrame, pl.DataFrame)):
                    return to_json_serializable(shell_result)
                return result
            except:
                return result

        tool_impl.__name__ = full_tool_name
        tool_impl.__doc__ = "\n".join(doc_lines)
        tool_impl.__signature__ = signature

        return tool_impl

    tool_func = create_tool_function()

    decorated_func = app.tool()(tool_func)

    setattr(app, f"_external_tool_{full_tool_name}", decorated_func)


def generate_tool_execution_code(
    mcp_name: str,
    mcp_config: dict[str, Any],
    tool_name: str,
    params: dict[str, Any],
    save_as: str,
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
    env_setup = "import os\nenv = os.environ.copy()\n"
    if "env" in mcp_config:
        for key, value in mcp_config["env"].items():
            if isinstance(value, str):
                if value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    env_setup += f"env['{key}'] = os.environ.get('{env_var}', '')\n"
                else:
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
    command={repr(mcp_config["command"])},
    args={repr(mcp_config.get("args", []))},
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
                        timeout=15.0
                    )
                except asyncio.TimeoutError:
                    raise Exception(f"Timeout calling tool {tool_name} on {mcp_name} (15s limit exceeded). The operation took too long, likely due to large file size or complex processing. Consider using alternative approaches like: 1) Reading specific file sections instead of entire file, 2) Using file streaming, 3) Filtering/limiting data before processing, or 4) Breaking large operations into smaller chunks.")
                
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
                
    except asyncio.TimeoutError as e:
        import logging
        logger = logging.getLogger(__name__)
        error_msg = f"Timeout error executing {mcp_name}.{tool_name}: Operation exceeded time limit"
        logger.error(f"{{error_msg}}: {{e}}")
        raise Exception(error_msg)
    except (ConnectionError, OSError, IOError) as e:
        import logging
        logger = logging.getLogger(__name__)
        error_msg = f"Connection error executing {mcp_name}.{tool_name}: Failed to connect or communicate with external MCP"
        logger.error(f"{{error_msg}}: {{e}}")
        raise Exception(error_msg)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        error_msg = f"Error executing {mcp_name}.{tool_name}: " + str(e)
        logger.error(error_msg)
        raise Exception(error_msg)

# Execute the async function with comprehensive error handling
try:
    _result = await _execute_external_tool()
except Exception as e:
    # Capture any exception and return as structured error
    error_details = {{
        "error": str(e),
        "error_type": type(e).__name__,
        "tool": "{tool_name}",
        "mcp": "{mcp_name}",
        "suggestion": "Try alternative approaches if this was a timeout or large file operation"
    }}
    _result = error_details
    print(f"External MCP tool execution failed: {{str(e)}}")

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


def convert_to_dataframe(data: Any) -> pd.DataFrame | Any:
    """Convert various data formats to DataFrame (polars preferred, pandas fallback).

    This function now uses polars for better performance when available,
    falling back to pandas for compatibility. The result may be either
    a polars or pandas DataFrame depending on availability and success.

    Args:
        data: Input data in various formats

    Returns:
        DataFrame: Polars DataFrame if available, otherwise pandas DataFrame

    """
    try:
        # Avoid blindly converting plain dicts (e.g. {"foo": "bar"}) into a
        # single-row DataFrame – that tends to break JSON-serialisation and makes
        # the response harder to consume programme-matically.  We only convert to
        # a DataFrame when the structure *looks* tabular (list/sequence of
        # mappings or sequences of equal length).

        _should_convert = True

        if isinstance(data, dict):
            # Heuristics:
            # 1. Every value is list-like OR
            # 2. Value count > 1 and at least one value is list-like.
            # Otherwise, keep as-is.
            list_like_values = [
                isinstance(v, (list, tuple)) for v in data.values()
            ]

            if not any(list_like_values):
                _should_convert = False

        if _should_convert:
            df = smart_create_dataframe(data, prefer_polars=True)
            return df
        else:
            return data

    except Exception as e:
        logger.warning(f"DataFrame conversion failed: {e}, returning original data")
        return data


_SHELL.push({"convert_to_dataframe": convert_to_dataframe})


if _external_mcps_available():
    logger.info("External MCPs configuration detected - registering external MCP tools")

    @app.tool()
    async def list_external_tools(server: str | None = None) -> dict[str, Any]:
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

            server_info = {"tools": []}

            for tool_name, tool_info in tools.items():
                tool_data = {
                    "name": tool_name,
                    "full_name": f"{mcp_name}_{tool_name}",
                    "description": tool_info.description,
                }

                if hasattr(tool_info, "inputSchema") and tool_info.inputSchema:
                    schema = tool_info.inputSchema
                    if isinstance(schema, dict) and "properties" in schema:
                        tool_data["parameters"] = {
                            "properties": schema["properties"],
                            "required": schema.get("required", []),
                        }

                server_info["tools"].append(tool_data)

            result[mcp_name] = server_info

        return result

else:
    logger.info(
        "External MCPs configuration not detected - external MCP tools will not be registered"
    )
