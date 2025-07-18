"""External MCP Tools for Sherlog MCP Server

This module provides dynamic integration with external MCP servers.
It discovers and registers tools from configured MCP servers at runtime.
"""

import asyncio
import inspect
import json
import os
from typing import Any, Optional

import pandas as pd
import polars as pl
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from sherlog_mcp.config import get_settings
from sherlog_mcp.dataframe_utils import smart_create_dataframe, to_json_serializable
from sherlog_mcp.ipython_shell_utils import run_code_in_shell
from fastmcp import Context
from sherlog_mcp.session import app, logger


EXTERNAL_TOOLS_REGISTRY: dict[str, dict[str, types.Tool]] = {}


async def auto_register_external_mcps():
    """Automatically discover and register all tools from configured external MCPs.

    This function is called during server startup to dynamically add tools
    from external MCP servers to the Sherlog MCP server.
    """
    settings = get_settings()

    print(settings.external_mcps)
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


def _create_parameter_from_schema(param_name: str, param_info: dict, is_required: bool = False) -> inspect.Parameter:
    """Create an inspect.Parameter from JSON schema property info."""
    param_type = param_info.get("type", "string")
    default_value = param_info.get("default", inspect.Parameter.empty)
    
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    
    annotation = type_mapping.get(param_type, Any)
    
    # Handle required vs optional parameters correctly
    if is_required:
        # Required parameters should not have defaults and should not be Optional
        return inspect.Parameter(
            param_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=inspect.Parameter.empty,
            annotation=annotation
        )
    else:
        # Optional parameters get Optional type annotation and None default
        if default_value == inspect.Parameter.empty:
            default_value = None
        annotation = Optional[annotation]
        return inspect.Parameter(
            param_name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=default_value,
            annotation=annotation
        )


def register_external_tool(
    mcp_name: str, tool_info: types.Tool, mcp_config: dict[str, Any]
):
    """Register a single external MCP tool with sherlog mcp.

    Args:
        mcp_name: Name of the MCP server
        tool_info: Tool information from the external MCP
        mcp_config: Configuration for connecting to the MCP

    """
    full_tool_name = f"external_{mcp_name}_{tool_info.name}"

    doc_lines = []
    if tool_info.description:
        doc_lines.append(tool_info.description)
    doc_lines.append(f"\n[External MCP: {mcp_name}]")
    doc_lines.append(f"[Original tool: {tool_info.name}]")

    parameters_required = []
    parameters_optional = []
    required_params = set()
    
    if hasattr(tool_info, "inputSchema") and tool_info.inputSchema:
        doc_lines.append("\nParameters:")
        schema = tool_info.inputSchema
        if isinstance(schema, dict) and "properties" in schema:
            required_params = set(schema.get("required", []))
            
            logger.debug(f"Tool {tool_info.name} schema: {schema}")
            logger.debug(f"Required params: {required_params}")
            
            for param_name, param_info in schema["properties"].items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                required = param_name in required_params
                req_str = " (required)" if required else " (optional)"
                doc_lines.append(
                    f"  {param_name}: {param_type}{req_str} - {param_desc}"
                )
                try:
                    param_obj = _create_parameter_from_schema(param_name, param_info, is_required=required)
                    if required:
                        parameters_required.append(param_obj)
                    else:
                        parameters_optional.append(param_obj)
                    logger.debug(f"Created parameter {param_name}: {param_obj}")
                except Exception as e:
                    logger.warning(f"Failed to create parameter {param_name}: {e}")

    # Merge parameters with required first, then optional
    parameters = parameters_required + parameters_optional

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
    doc_lines.append(
        "\nResults persist as '{save_as}'."
    )
    doc_lines.append(
        "\nExamples"
    )
    doc_lines.append(
        "--------"
    )
    doc_lines.append(
        "After calling this tool with save_as=\"result\":"
    )
    doc_lines.append(
        ""
    )
    doc_lines.append(
        "# View the result (format depends on the tool's return type)"
    )
    doc_lines.append(
        '>>> execute_python_code("result")'
    )
    doc_lines.append(
        ""
    )
    doc_lines.append(
        "# If result is a DataFrame:"
    )
    doc_lines.append(
        '>>> execute_python_code("result.head()")'
    )
    doc_lines.append(
        '>>> execute_python_code("result.info()")'
    )
    doc_lines.append(
        '>>> execute_python_code("result.columns.tolist()")'
    )
    doc_lines.append(
        ""
    )
    doc_lines.append(
        "# If result is a string/text:"
    )
    doc_lines.append(
        '>>> execute_python_code("print(result[:500])")  # First 500 characters'
    )
    doc_lines.append(
        '>>> execute_python_code("len(result.splitlines())")  # Count lines'
    )
    doc_lines.append(
        ""
    )
    doc_lines.append(
        "# If result is a list/dict:"
    )
    doc_lines.append(
        '>>> execute_python_code("type(result)")  # Check data type'
    )
    doc_lines.append(
        '>>> execute_python_code("len(result)")  # Check length'
    )
    doc_lines.append(
        '>>> execute_python_code("import json; print(json.dumps(result, indent=2)[:1000])")  # Pretty print'
    )

    signature = inspect.Signature(parameters)
    logger.debug(f"Created signature for {tool_info.name}: {signature}")

    ######################################################################
    # Build the *internal* implementation that performs the heavy lifting
    ######################################################################

    async def _internal_tool_impl(**_kwargs):
        """Internal helper that executes the external MCP call.
        Accepts **kwargs with the full parameter set generated below."""
        try:
            bound_args = signature.bind_partial(**_kwargs)
            bound_args.apply_defaults()
        except TypeError as e:
            raise ValueError(f"Invalid arguments for {tool_info.name}: {e}")

        call_params = dict(bound_args.arguments)
        save_as = call_params.pop("save_as", f"{full_tool_name}_result")

        # Validate required params
        missing_required = [
            p for p in required_params if call_params.get(p, None) is None
        ]
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

        try:
            from sherlog_mcp.middleware.session_middleware import get_session_shell

            session_id = "default"
            shell = get_session_shell(session_id)
            if not shell:
                raise RuntimeError(f"No shell found for session {session_id}")

            execution_result = await run_code_in_shell(code, shell, session_id)

            if execution_result and hasattr(execution_result, "error_in_exec") and execution_result.error_in_exec:
                return {
                    "error": str(execution_result.error_in_exec),
                    "error_type": type(execution_result.error_in_exec).__name__,
                    "tool": tool_info.name,
                    "mcp": mcp_name,
                    "suggestion": "Tool execution failed",
                }

            shell_result = shell.user_ns.get(save_as)
            if isinstance(shell_result, (pd.DataFrame, pl.DataFrame)):
                return to_json_serializable(shell_result)
            elif isinstance(shell_result, dict) and "error" in shell_result:
                return shell_result
            else:
                return execution_result.result if execution_result else None
        except Exception as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "tool": tool_info.name,
                "mcp": mcp_name,
                "suggestion": "Tool execution failed",
            }

    ######################################################################
    # Dynamically create a wrapper with the REAL signature so that FastMCP
    # sees the correct parameter list when decorating.
    ######################################################################

    def _build_wrapper_fn(sig: inspect.Signature):
        """Return an async wrapper function that forwards to _internal_tool_impl.
        The wrapper is built with `exec` so its code object contains the
        explicit parameter list that FastMCP needs."""

        # Build parameter string (without annotations to avoid import issues)
        parts = []
        star_added = False
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.KEYWORD_ONLY and not star_added:
                parts.append("*")
                star_added = True
            param_src = p.name
            if p.default is not inspect.Parameter.empty:
                param_src += f"={repr(p.default)}"
            parts.append(param_src)
        params_src = ", ".join(parts)

        wrapper_src = (
            "async def _wrapper(" + params_src + "):\n"
            "    return await _internal_tool_impl(**locals())\n"
        )

        _globals: dict[str, Any] = {
            "_internal_tool_impl": _internal_tool_impl,
            "await": None,  # placeholder for exec safety
        }
        _locals: dict[str, Any] = {}
        exec(wrapper_src, _globals, _locals)
        return _locals["_wrapper"]

    tool_func = _build_wrapper_fn(signature)

    tool_func.__name__ = full_tool_name
    tool_func.__doc__ = "\n".join(doc_lines)
    # The wrapper already has the correct signature

    # Register with FastMCP
    logger.debug(f"About to register tool {full_tool_name} with real signature: {inspect.signature(tool_func)}")
    try:
        decorated_func = app.tool()(tool_func)
        logger.debug(f"Successfully registered {full_tool_name}")
    except Exception as e:
        logger.error(f"Failed during app.tool() decoration for {tool_info.name}: {e}")
        logger.error(f"Tool signature was: {inspect.signature(tool_func)}")
        raise

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
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import pandas as pd
import numpy as np
import logging

# Configure logging to use stderr
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

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
                
    except asyncio.CancelledError as e:
        import sys
        error_msg = f"Operation cancelled for {mcp_name}.{tool_name}: Task was cancelled due to timeout or external interruption"
        print(error_msg, file=sys.stderr)
        raise Exception(error_msg)
    except asyncio.TimeoutError as e:
        import sys
        error_msg = f"Timeout error executing {mcp_name}.{tool_name}: Operation exceeded time limit"
        print(error_msg, file=sys.stderr)
        raise Exception(error_msg)
    except (ConnectionError, OSError, IOError) as e:
        import sys
        error_msg = f"Connection error executing {mcp_name}.{tool_name}: Failed to connect or communicate with external MCP"
        print(f"{{error_msg}}: {{e}}", file=sys.stderr)
        raise Exception(error_msg)
    except Exception as e:
        import sys
        error_msg = f"Error executing {mcp_name}.{tool_name}: " + str(e)
        print(error_msg, file=sys.stderr)
        raise Exception(error_msg)

# Execute the async function with comprehensive error handling
try:
    _result = await _execute_external_tool()
except asyncio.CancelledError as e:
    # Handle cancellation specifically 
    error_details = {{
        "error": "Operation was cancelled due to timeout or interruption",
        "error_type": "CancelledError", 
        "tool": "{tool_name}",
        "mcp": "{mcp_name}",
        "suggestion": "The operation was cancelled, likely due to timeout. Try breaking large operations into smaller chunks or using alternative approaches."
    }}
    _result = error_details
    import sys
    print(f"External MCP tool execution cancelled: Operation timeout or interruption", file=sys.stderr)
except Exception as e:
    # Capture any other exception and return as structured error
    error_details = {{
        "error": str(e),
        "error_type": type(e).__name__,
        "tool": "{tool_name}",
        "mcp": "{mcp_name}",
        "suggestion": "Try alternative approaches if this was a timeout or large file operation"
    }}
    _result = error_details
    import sys
    print(f"External MCP tool execution failed: {{str(e)}}", file=sys.stderr)

# Convert to DataFrame if possible
{save_as} = convert_to_dataframe(_result)

# Store result info in a variable instead of printing
_result_info = f"Result stored in '{save_as}'"
if isinstance({save_as}, pd.DataFrame):
    _result_info += f"\\nDataFrame shape: {{{save_as}.shape}}"
    _result_info += f"\\nColumns: {{{save_as}.columns.tolist()}}"

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
        _should_convert = True

        if isinstance(data, dict):
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
