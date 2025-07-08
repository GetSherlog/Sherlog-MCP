"""CLI Discovery and Execution Tools

This module provides tools for discovering installed CLI tools and packages,
and executing CLI commands with DataFrame output support.
"""

from typing import Any
import pandas as pd
from sherlog_mcp.tools.utilities import return_result
from sherlog_mcp.ipython_shell_utils import run_code_in_shell
from fastmcp import Context
from sherlog_mcp.session import app
from .pypi_core import (
    PyPIClient,
    InvalidPackageNameError,
    PackageNotFoundError,
    NetworkError,
    PyPIError,
)
from sherlog_mcp.middleware.session_middleware import get_session_shell
import logging

logger = logging.getLogger(__name__)

def format_package_info(package_data: dict[str, Any]) -> dict[str, Any]:
    """Format package information for MCP response.

    Args:
        package_data: Raw package data from PyPI API

    Returns:
        Formatted package information
    """
    info = package_data.get("info", {})

    formatted = {
        "name": info.get("name", ""),
        "version": info.get("version", ""),
        "summary": info.get("summary", ""),
        "description": info.get("description", "")[:500] + "..."
        if len(info.get("description", "")) > 500
        else info.get("description", ""),
        "author": info.get("author", ""),
        "author_email": info.get("author_email", ""),
        "maintainer": info.get("maintainer", ""),
        "maintainer_email": info.get("maintainer_email", ""),
        "license": info.get("license", ""),
        "home_page": info.get("home_page", ""),
        "project_url": info.get("project_url", ""),
        "download_url": info.get("download_url", ""),
        "requires_python": info.get("requires_python", ""),
        "platform": info.get("platform", ""),
        "keywords": info.get("keywords", ""),
        "classifiers": info.get("classifiers", []),
        "requires_dist": info.get("requires_dist", []),
        "project_urls": info.get("project_urls", {}),
    }

    releases = package_data.get("releases", {})
    formatted["total_versions"] = len(releases)
    formatted["available_versions"] = list(releases.keys())[-10:]  # Last 10 versions

    if "urls" in package_data:
        urls = package_data["urls"]
        if urls:
            formatted["download_info"] = {
                "files_count": len(urls),
                "file_types": list({url.get("packagetype", "") for url in urls}),
                "python_versions": list(
                    {
                        url.get("python_version", "")
                        for url in urls
                        if url.get("python_version")
                    }
                ),
            }

    return formatted


async def query_package_info(package_name: str) -> dict[str, Any]:
    """Query comprehensive package information from PyPI.

    Args:
        package_name: Name of the package to query

    Returns:
        Formatted package information

    Raises:
        InvalidPackageNameError: If package name is invalid
        PackageNotFoundError: If package is not found
        NetworkError: For network-related errors
    """
    if not package_name or not package_name.strip():
        raise InvalidPackageNameError(package_name)

    try:
        async with PyPIClient() as client:
            package_data = await client.get_package_info(package_name)
            return format_package_info(package_data)
    except PyPIError:
        raise
    except Exception as e:
        raise NetworkError(f"Failed to query package information: {e}", e) from e


async def _search_pypi_impl(package_name: str) -> pd.DataFrame:
    try:
        result = await query_package_info(package_name)
        return pd.DataFrame([result])
    except (InvalidPackageNameError, PackageNotFoundError, NetworkError) as e:
        error =  {
            "error": str(e),
            "error_type": type(e).__name__,
            "package_name": package_name,
        }
        return pd.DataFrame([error])
    except Exception as e:
        error = {
            "error": f"Unexpected error: {e}",
            "error_type": "UnexpectedError",
            "package_name": package_name,
        }
        return pd.DataFrame([error])


@app.tool()
async def call_cli(
    command: str,
    save_as: str,
    ctx: Context,
) -> dict:
    """
    Execute a CLI command using IPython's ! syntax and save the result.
    
    Args:
        command: The CLI command to execute
        save_as: Variable name to save the result in IPython namespace
        
    Returns:
        dict: Response stating that if the command was successful and saved to the IPython namespace.
        You can inspect the result `save_as` and work with it as you would with any other variable.

    If a command fails, it usually means that the command is not installed.
    You can install it with an apt install command.
    For example, if the command is "gh", you can install it with:
    >>> call_cli("!apt install gh")

    When in doubt on how to install a command, just search for it online. 
        
    Examples:
        - call_cli("gh repo list --json name,description", save_as="repos")
        - call_cli("ls -la", save_as="files")
        - call_cli("docker ps")
        
    After calling this tool:
        
    # If save_as was provided, access the result
    >>> execute_python_code("repos")  # Raw command output
    
    # Access command output attributes
    >>> execute_python_code("files.s")  # As string
    >>> execute_python_code("files.n")  # With line numbers
    """

    # For multiline commands or commands with special characters, use subprocess
    if '\n' in command or '<<' in command or '$(' in command:
        code = f"""
import subprocess
import shlex

try:
    # Execute the command using subprocess for better multiline support
    result = subprocess.run(
        {repr(command)},
        shell=True,
        capture_output=True,
        text=True
    )
    
    # Create a list of output lines similar to IPython's ! syntax
    output_lines = result.stdout.splitlines() if result.stdout else []
    if result.stderr:
        output_lines.extend(result.stderr.splitlines())
    
    # Store result in a format similar to IPython's SList
    {save_as} = output_lines
    {save_as}
except Exception as e:
    {save_as} = [f"Error executing command: {{e}}"]
    {save_as}
"""
    else:
        # For simple single-line commands, use IPython's ! syntax
        code = f"{save_as} = !{command}\n{save_as}"
    
    session_id = ctx.session_id or "default"
    shell = get_session_shell(session_id)
    if not shell:
        raise RuntimeError(f"No shell found for session {session_id}")
    execution_result = await run_code_in_shell(code, shell, session_id)
    logger.info(f"Executing command: {command}")
    return return_result(code, execution_result, command, save_as)


@app.tool()
async def search_pypi(
    query: str,
    *, 
    save_as: str,
    ctx: Context,
) -> dict:
    """
    Search PyPI for Python packages.
    
    Args:
        query: Search query for package names or descriptions
        save_as: Variable name to save results in IPython shell
        
    Returns:
        dict: Response with search results
        
    Examples
    --------
    After calling this tool with save_as="search_results":
    
    # View search results
    >>> execute_python_code("search_results")
    
    # Get package names
    >>> execute_python_code("search_results['name'].tolist()")
    """
    code = f'{save_as} = _search_pypi_impl("{query}")\n{save_as}'
    session_id = ctx.session_id or "default"
    shell = get_session_shell(session_id)
    if not shell:
        raise RuntimeError(f"No shell found for session {session_id}")
    execution_result = await run_code_in_shell(code, shell, session_id)
    return return_result(code, execution_result, query, save_as)

def _query_apt_package_status_impl(package: str) -> pd.DataFrame:
    import subprocess
    try:
        def run(cmd: str):
            return subprocess.run(cmd, shell=True, capture_output=True, text=True)

        installed_res = run(f"dpkg -s {package}")
        installed = "installed" if installed_res.returncode == 0 else "not installed"

        upgradable_res = run(
            f"apt list --upgradable 2>/dev/null | grep '^{package}/' || true"
        )
        upgradable = bool(upgradable_res.stdout.strip())

        available_res = run(f"apt-cache show {package}")
        available = "available" if available_res.stdout.strip() else "not available"

        data = {
            "package": package,
            "installed": installed,
            "upgradable": upgradable,
            "available": available,
        }
        return pd.DataFrame([data])
    except Exception as e:
        error = {
            "error": str(e),
            "error_type": type(e).__name__,
            "package": package,
        }
        return pd.DataFrame([error])


@app.tool()
async def query_apt_package_status(package: str, *, save_as: str, ctx: Context) -> dict:
    """
    Query apt package status (installed, upgradable, available).

    Args:
        package: Name of the package to query
        save_as: Variable name to save results in IPython shell

    Returns:
        dict: Response with package status information
    """
    code = f'{save_as} = _query_apt_package_status_impl("{package}")\n{save_as}'

    session_id = ctx.session_id or "default"
    shell = get_session_shell(session_id)
    if not shell:
        raise RuntimeError(f"No shell found for session {session_id}")
    execution_result = await run_code_in_shell(code, shell, session_id)
    return return_result(code, execution_result, package, save_as)