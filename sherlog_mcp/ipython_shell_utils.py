import contextlib
import io
from typing import Any

from IPython.core.interactiveshell import InteractiveShell
from fastmcp import Context

from sherlog_mcp.session import app
import fnmatch
import pandas as pd
import polars as pl
import os
import sys
import traceback
import inspect
import numpy as np
from sherlog_mcp.dataframe_utils import to_json_serializable

def get_session_shell(session_id):
    from sherlog_mcp.middleware.session_middleware import get_session_shell as _get_session_shell
    return _get_session_shell(session_id)


class SmartMemoryManager:
    """Automatically manages IPython session memory to prevent bloat."""
    
    def __init__(self):
        from sherlog_mcp.config import get_settings
        settings = get_settings()
        self.execution_counts = {}  # session_id -> count
        self.last_reset_counts = {}  # session_id -> count
        self.reset_threshold = settings.auto_reset_threshold
        self.auto_reset_enabled = settings.auto_reset_enabled
        
    def should_reset(self, session_id: str, shell: InteractiveShell) -> bool:
        """Check if we should reset based on execution count and presence of DataFrames."""
        if session_id not in self.execution_counts:
            self.execution_counts[session_id] = 0
            self.last_reset_counts[session_id] = 0
            
        self.execution_counts[session_id] += 1
        
        if not self.auto_reset_enabled:
            return False
        
        executions_since_reset = self.execution_counts[session_id] - self.last_reset_counts[session_id]
        
        if executions_since_reset >= self.reset_threshold:
            has_dataframes = any(
                isinstance(obj, (pd.DataFrame, pl.DataFrame)) 
                for obj in shell.user_ns.values()
            )
            
            if has_dataframes:
                self.last_reset_counts[session_id] = self.execution_counts[session_id]
                return True
                
        return False
    
    def reset(self, shell: InteractiveShell):
        """Smart reset - preserves imports and recent DataFrames."""
        recent_dfs = {}
        for name, obj in shell.user_ns.items():
            if isinstance(obj, (pd.DataFrame, pl.DataFrame)) and not name.startswith('_'):
                if len(recent_dfs) < 3:
                    recent_dfs[name] = obj
        
        imports = {k: v for k, v in shell.user_ns.items() 
                   if hasattr(v, '__module__') and not k.startswith('_')}
        
        shell.reset()
        
        shell.user_ns.update(imports)
        shell.user_ns.update(recent_dfs)
        
        shell.run_cell("import pandas as pd\nimport numpy as np\nimport polars as pl")


_SMART_MANAGER = SmartMemoryManager()


async def run_code_in_shell(code: str, shell: InteractiveShell, session_id: str = "default"):
    """Execute *code* asynchronously in the given IPython shell and return the ExecutionResult."""
    if _SMART_MANAGER.should_reset(session_id, shell):
        _SMART_MANAGER.reset(shell)

    execution_result = await shell.run_cell_async(code, silent=True)

    return execution_result


@app.tool()
async def execute_python_code(code: str, ctx: Context):
    """Executes a given string of Python code in the underlying IPython interactive shell.

    Executes Python code in a persistent IPython session where all variables
    from previous tool calls are available. Use list_dataframes() to see available
    data or list_shell_variables() for all variables.

    This tool allows for direct execution of arbitrary Python code, including
    defining variables, calling functions, or running any valid Python statements.
    The code is run in the same IPython shell instance used by other tools,
    allowing for state sharing (variables defined in one call can be used in subsequent calls).

    Parameters
    ----------
    code : str
        A string containing the Python code to be executed.
        For example, "x = 10+5" or "print(\\'Hello, world!\\')" or
        "my_variable = some_function_defined_elsewhere()".

    Returns
    -------
    Any
        The result of the last expression in the executed code. If the code
        does not produce a result (e.g., an assignment statement), it might
        return None or as per IPython's `run_cell_async` behavior for such cases.
        Specifically, it returns `execution_result.result` from IPython's
        `ExecutionResult` object.

    Examples
    --------
    # Define a variable
    >>> execute_python_code(code="my_var = 42")
    # Use the defined variable
    >>> execute_python_code(code="print(my_var * 2)")
    # Output: 84

    # Execute a multi-line script
    >>> script = \'\'\'
    ... import math
    ... def calculate_circle_area(radius):
    ...     return math.pi * radius ** 2
    ... area = calculate_circle_area(5)
    ... area
    ... \'\'\'
    >>> execute_python_code(code=script)
    # Output: 78.53981633974483

    See Also
    --------
    IPython.core.interactiveshell.InteractiveShell.run_cell_async
    run_code_in_shell (internal utility called by this tool)

    """
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    from sherlog_mcp.config import get_settings
    settings = get_settings()
    MAX_OUTPUT_SIZE = settings.max_output_size

    with (
        contextlib.redirect_stdout(stdout_buffer),
        contextlib.redirect_stderr(stderr_buffer),
    ):
        try:
            session_id = ctx.session_id or "default"
            shell = get_session_shell(session_id)
            if not shell:
                raise RuntimeError(f"No shell found for session {session_id}")
            result = await run_code_in_shell(code, shell, session_id)
        except Exception as e:
            result = None

    stdout_value = stdout_buffer.getvalue()
    stderr_value = stderr_buffer.getvalue()
    
    if len(stdout_value) > MAX_OUTPUT_SIZE:
        stdout_value = stdout_value[:MAX_OUTPUT_SIZE] + "\n... (output truncated)"
    if len(stderr_value) > MAX_OUTPUT_SIZE:
        stderr_value = stderr_value[:MAX_OUTPUT_SIZE] + "\n... (output truncated)"

    execution_details_dict = {}

    if result is not None:
        if not result.error_before_exec and not result.error_in_exec:
            execution_details_dict["result"] = to_json_serializable(result.result)

        if result.error_before_exec:
            execution_details_dict["error_before_exec"] = str(result.error_before_exec)
            execution_details_dict["success"] = False

        if result.error_in_exec:
            error_type = type(result.error_in_exec).__name__
            error_msg = str(result.error_in_exec)
            execution_details_dict["error_in_exec"] = f"{error_type}: {error_msg}"
            execution_details_dict["success"] = False
            try:
                if sys.exc_info()[0] is not None:
                    tb_lines = traceback.format_exc()
                    execution_details_dict["traceback"] = tb_lines[:8192]
            except Exception:
                pass
        else:
            execution_details_dict["success"] = True

    if stdout_value:
        execution_details_dict["stdout"] = stdout_value.rstrip()

    if stderr_value:
        execution_details_dict["stderr"] = stderr_value.rstrip()

    return execution_details_dict


@app.tool()
async def list_shell_variables(ctx: Context) -> list[str]:
    """Lists variable names in the current IPython user namespace.

    Tries to exclude common IPython internal variables (e.g., 'In', 'Out', 'exit', 'quit', 'get_ipython')
    and variables starting with an underscore unless they are common history accessors.
    Special underscore variables like '_', '__', '___' (output history) and
    '_i', '_ii', '_iii' (input history) are included if present.

    Returns
    -------
    list[str]
        A sorted list of identified user variable names. To get the value of a variable, use the `inspect_shell_object` tool.

    """
    user_vars = []
    system_variables = {
        "In",
        "Out",
        "exit",
        "quit",
        "get_ipython",
        "_ih",
        "_oh",
        "_dh",
        "_sh",
        "_ip",
    }

    session_id = ctx.session_id or "default"
    shell = get_session_shell(session_id)
    if not shell or shell.user_ns is None:
        return []

    for name in shell.user_ns.keys():
        if name in system_variables:
            continue
        if (
            name.startswith("_")
            and name not in {"_", "__", "___", "_i", "_ii", "_iii"}
            and not name.startswith("_i")
        ):
            continue
        user_vars.append(name)
    return sorted(list(set(user_vars)))


@app.tool()
async def inspect_shell_object(object_name: str, ctx: Context, detail_level: int = 0) -> str:
    """Provides detailed information about an object in the IPython shell by its name.
    Uses IPython's object inspector.

    Parameters
    ----------
    object_name : str
        The name of the variable/object in the shell to inspect.
    detail_level : int, optional
        Detail level for inspection (0, 1, or 2).
        0: basic info (type, string representation).
        1: adds docstring.
        2: adds source code if available.
        Defaults to 0.

    Returns
    -------
    str
        A string containing the inspection details.
        Returns an error message if the object is not found or if an error occurs during inspection.

    """
    session_id = ctx.session_id or "default"
    shell = get_session_shell(session_id)
    if not shell or shell.user_ns is None or object_name not in shell.user_ns:
        return f"Error: Object '{object_name}' not found in the shell namespace."
    try:
        actual_detail_level = min(max(detail_level, 0), 2)
        return shell.object_inspect_text(object_name, detail_level=actual_detail_level)
    except Exception as e:
        return f"Error during inspection of '{object_name}': {str(e)}"


@app.tool()
async def get_shell_history(ctx: Context, range_str: str = "", raw: bool = False) -> str:
    """Retrieves lines from the IPython shell's input history.

    Uses IPython's `extract_input_lines` method. The `range_str` defines which lines to retrieve.
    Examples for `range_str`:
    - "1-5": Lines 1 through 5 of the current session.
    - "~2/1-5": Lines 1 through 5 of the second-to-last session.
    - "6": Line 6 of the current session.
    - "": (Default) All lines of the current session except the last executed one.
    - "~10:": All lines starting from the 10th line of the last session.
    - ":5": Lines up to 5 of current session.

    Parameters
    ----------
    range_str : str, optional
        A string specifying the history slices to retrieve, by default "" (all current session history, except last).
        The syntax is based on IPython's history access (%history magic).
    raw : bool, optional
        If True, retrieves the raw, untransformed input history. Defaults to False.

    Returns
    -------
    str
        A string containing the requested input history lines, separated by newlines.
        Returns an error message if history retrieval fails.

    """
    try:
        session_id = ctx.session_id or "default"
        shell = get_session_shell(session_id)
        if not shell:
            return "Error: No shell found for session"
        history_lines = shell.extract_input_lines(range_str=range_str, raw=raw)
        return history_lines
    except Exception as e:
        return f"Error retrieving shell history for range '{range_str}' (raw={raw}): {str(e)}"


@app.tool()
async def run_shell_magic(magic_name: str, line: str, ctx: Context, cell: str | None = None):
    """Executes an IPython magic command in the shell.

    Allows execution of both line magics (e.g., %ls -l) and cell magics (e.g., %%timeit code...).

    Parameters
    ----------
    magic_name : str
        The name of the magic command (e.g., "ls", "timeit", "writefile") WITHOUT the leading '%' or '%%'.
    line : str
        The argument string for the magic command. For line magics, this is the entire line after the magic name.
        For cell magics, this is the line immediately following the `%%magic_name` directive.
        Can be an empty string if the magic command takes no arguments on its first line.
    cell : str, optional
        The body of a cell magic (the code block below `%%magic_name line`).
        If None or an empty string, the command is treated as a line magic.
        If provided, it's treated as a cell magic.

    Returns
    -------
    Any
        The result of the magic command execution, if any. Behavior varies depending on the magic command.
        May return None, text output, or other objects. In case of errors, an error message string is returned.

    Examples
    --------
    # Line magic example: list files
    >>> run_shell_magic(magic_name="ls", line="-la")

    # Cell magic example: time a piece of code
    >>> run_shell_magic(magic_name="timeit", line="-n 10", cell="sum(range(100))")

    # Magic that doesn't produce a return value directly to python but has side effects (e.g. writing a file)
    >>> run_shell_magic(magic_name="writefile", line="my_test_file.txt", cell="This is a test.")

    """
    try:
        if cell is not None and cell.strip() != "":
            session_id = ctx.session_id or "default"
            shell = get_session_shell(session_id)
            if not shell:
                return "Error: No shell found for session"
            return shell.run_cell_magic(magic_name, line, cell)
        else:
            session_id = ctx.session_id or "default"
            shell = get_session_shell(session_id)
            if not shell:
                return "Error: No shell found for session"
            return shell.run_line_magic(magic_name, line)
    except Exception as e:
        error_type = type(e).__name__
        return f"Error executing magic command '{magic_name}' (line='{line}', cell present: {cell is not None}): {error_type}: {str(e)}"


@app.tool()
async def install_package(package_spec: str, ctx: Context, upgrade: bool = False):
    """Installs a Python package using uv within the IPython shell session.

    This tool allows the LLM to install packages dynamically using IPython's magic commands.
    The package will be installed in the same environment where the IPython shell is running
    and will be immediately available for import in subsequent code executions.

    Parameters
    ----------
    package_spec : str
        The package specification to install. Can be:
        - A simple package name: "requests"
        - A package with version: "requests==2.31.0"
        - A package with version constraints: "requests>=2.30.0"
        - A git repository: "git+https://github.com/user/repo.git"
        - A local path: "/path/to/package"
        - Multiple packages: "requests numpy pandas"
    upgrade : bool, optional
        Whether to upgrade the package if it's already installed. Defaults to False.

    Returns
    -------
    dict
        A dictionary containing:
        - "success": bool indicating if installation succeeded
        - "output": str with installation output or error message
        - "packages_requested": list of package names that were requested for installation

    Examples
    --------
    # Install a single package
    >>> install_package("requests")

    # Install with version constraint
    >>> install_package("numpy>=1.20.0")

    # Install and upgrade if already present
    >>> install_package("matplotlib", upgrade=True)

    # Install from git repository
    >>> install_package("git+https://github.com/user/repo.git@main")

    """
    try:
        pip_args = []

        if upgrade:
            pip_args.append("--upgrade")
        pip_args.append(package_spec)

        pip_command_line = " ".join(pip_args)

        session_id = ctx.session_id or "default"
        shell = get_session_shell(session_id)
        if not shell:
            return {
                "success": False,
                "output": "Error: No shell found for session",
                "packages_requested": package_spec.split(),
            }
        magic_result = shell.run_line_magic("pip", f"install {pip_command_line}")

        packages_requested = []
        for pkg in package_spec.split():
            base_name = (
                pkg.split("==")[0]
                .split(">=")[0]
                .split("<=")[0]
                .split(">")[0]
                .split("<")[0]
                .split("@")[0]
            )
            if not base_name.startswith("git+"):
                packages_requested.append(base_name)
            else:
                if ".git" in base_name:
                    repo_name = base_name.split("/")[-1].replace(".git", "")
                    packages_requested.append(repo_name)

        return {
            "success": True,
            "output": str(magic_result)
            if magic_result
            else "Package installation completed successfully",
            "packages_requested": packages_requested,
        }

    except Exception as e:
        error_msg = str(e)
        if "No module named" in error_msg:
            error_msg += "\nNote: Package may need to be installed with a different name or from a different source."
        elif "Permission denied" in error_msg:
            error_msg += "\nNote: Installation may require different permissions in this environment."

        return {
            "success": False,
            "output": f"Installation failed: {error_msg}",
            "packages_requested": package_spec.split(),
        }


@app.tool()
async def get_completions(text: str, ctx: Context, cursor_pos: int | None = None) -> dict[str, Any]:
    """Get code completions at cursor position to help LLM understand available methods/attributes.

    This tool provides intelligent code completion suggestions that can help the LLM
    understand what methods, attributes, or variables are available in the current context.

    Parameters
    ----------
    text : str
        The code text for which to get completions
    cursor_pos : int, optional
        Position of cursor in the text. If None, defaults to end of text.

    Returns
    -------
    dict
        Dictionary containing:
        - "text": the actual text that was completed
        - "matches": list of possible completions
        - "cursor_start": position where completion starts
        - "cursor_end": position where completion ends

    Examples
    --------
    >>> get_completions("import o")
    {'text': 'o', 'matches': ['os', 'operator', 'optparse', ...], ...}

    >>> get_completions("np.arr")  # after importing numpy as np
    {'text': 'arr', 'matches': ['array', 'array_equal', 'array_split', ...], ...}

    """
    try:
        if cursor_pos is None:
            cursor_pos = len(text)

        session_id = ctx.session_id or "default"
        shell = get_session_shell(session_id)
        if not shell:
            return {
                "text": "",
                "matches": [],
                "cursor_start": cursor_pos or 0,
                "cursor_end": cursor_pos or 0,
                "total_matches": 0,
                "error": "No shell found for session",
            }
        completed_text, matches = shell.complete(text, cursor_pos=cursor_pos)

        if not isinstance(completed_text, str):
            completed_text = str(completed_text) if completed_text is not None else ""
        
        if not isinstance(matches, list):
            matches = []

        cursor_start = cursor_pos - len(completed_text)
        cursor_end = cursor_pos

        return {
            "text": completed_text,
            "matches": matches,
            "cursor_start": cursor_start,
            "cursor_end": cursor_end,
            "total_matches": len(matches) if hasattr(matches, '__len__') else 0,
        }
    except Exception as e:
        return {
            "text": "",
            "matches": [],
            "cursor_start": cursor_pos or 0,
            "cursor_end": cursor_pos or 0,
            "total_matches": 0,
            "error": str(e),
        }


@app.tool()
async def get_function_signature(func_name: str, ctx: Context) -> dict[str, Any]:
    """Get function signature and docstring to help LLM generate correct function calls.

    This tool provides detailed information about function signatures, parameters,
    and documentation, which helps the LLM understand how to correctly call functions.

    Parameters
    ----------
    func_name : str
        Name of the function/method/class to inspect

    Returns
    -------
    dict
        Dictionary containing function information:
        - "signature": function signature string
        - "docstring": function documentation
        - "type": type of the object (function, method, class, etc.)
        - "module": module where object is defined
        - "file": file where object is defined (if available)

    Examples
    --------
    >>> get_function_signature("print")
    {'signature': 'print(*args, sep=...', 'docstring': 'print(value, ..., sep=...', ...}

    >>> get_function_signature("pandas.DataFrame")
    {'signature': 'DataFrame(data=None, index=None, ...', 'docstring': 'Two-dimensional...', ...}

    """
    try:
        session_id = ctx.session_id or "default"
        shell = get_session_shell(session_id)
        if not shell:
            return {"error": "No shell found for session"}
        info = shell.object_inspect(func_name, detail_level=1)

        if not info:
            return {"error": f"Object '{func_name}' not found"}

        return {
            "signature": info.get("definition", ""),
            "docstring": info.get("docstring", ""),
            "type": info.get("type_name", ""),
            "module": info.get("namespace", ""),
            "file": info.get("file", ""),
            "class_docstring": info.get("class_docstring", ""),
            "init_docstring": info.get("init_docstring", ""),
            "call_def": info.get("call_def", ""),
            "call_docstring": info.get("call_docstring", ""),
        }
    except Exception as e:
        return {"error": str(e)}


@app.tool()
async def get_namespace_info(ctx: Context) -> dict[str, Any]:
    """Get information about current namespaces to help LLM understand scope.

    This tool provides insight into what variables, functions, and objects are
    currently available in different namespaces, helping the LLM understand
    the current execution context.

    Returns
    -------
    dict
        Dictionary with namespace information:
        - "user_variables": list of user-defined variable names
        - "builtin_names": list of available builtin names
        - "imported_modules": list of imported module names
        - "total_user_objects": count of objects in user namespace

    Examples
    --------
    >>> get_namespace_info()
    {'user_variables': ['x', 'df', 'my_func'], 'builtin_names': ['print', 'len', ...], ...}

    """
    try:
        user_vars = []
        system_variables = {
            "In",
            "Out",
            "exit",
            "quit",
            "get_ipython",
            "_ih",
            "_oh",
            "_dh",
            "_sh",
            "_ip",
        }

        session_id = ctx.session_id or "default"
        shell = get_session_shell(session_id)
        if not shell:
            return {"error": "No shell found for session"}
            
        if shell.user_ns:
            for name in shell.user_ns.keys():
                if name in system_variables:
                    continue
                if (
                    name.startswith("_")
                    and name not in {"_", "__", "___", "_i", "_ii", "_iii"}
                    and not name.startswith("_i")
                ):
                    continue
                user_vars.append(name)

        builtin_names = []
        try:
            import builtins

            builtin_names = [
                name for name in dir(builtins) if not name.startswith("_")
            ][:50]
        except:
            builtin_names = [
                "print",
                "len",
                "str",
                "int",
                "float",
                "list",
                "dict",
                "tuple",
                "set",
            ]

        imported_modules = []
        if shell.user_ns:
            for name, obj in shell.user_ns.items():
                if hasattr(obj, "__file__") and hasattr(obj, "__name__"):
                    if not name.startswith("_"):
                        imported_modules.append(name)

        return {
            "user_variables": sorted(user_vars),
            "builtin_names": sorted(builtin_names),
            "imported_modules": sorted(imported_modules),
            "total_user_objects": len(user_vars),
        }
    except Exception as e:
        return {"error": str(e)}


@app.tool()
async def get_object_source(object_name: str, ctx: Context) -> dict[str, Any]:
    """Get source code of functions/classes to help LLM understand implementation patterns.

    This tool retrieves the actual source code of functions, methods, and classes,
    which helps the LLM understand implementation patterns and coding styles.

    Parameters
    ----------
    object_name : str
        Name of the object to get source code for

    Returns
    -------
    dict
        Dictionary containing:
        - "source": source code string
        - "file": file where object is defined
        - "line_number": line number where object starts
        - "type": type of the object

    Examples
    --------
    >>> get_object_source("my_function")
    {'source': 'def my_function(x):\n    return x * 2', 'file': '<ipython-input-1>', ...}

    """
    try:
        session_id = ctx.session_id or "default"
        shell = get_session_shell(session_id)
        if not shell:
            return {"error": "No shell found for session"}
        info = shell.object_inspect(object_name, detail_level=2)

        if not info:
            return {"error": f"Object '{object_name}' not found"}

        return {
            "source": info.get("source", ""),
            "file": info.get("file", ""),
            "line_number": info.get("line_number", ""),
            "type": info.get("type_name", ""),
            "definition": info.get("definition", ""),
            "docstring": info.get("docstring", ""),
        }
    except Exception as e:
        return {"error": str(e)}


@app.tool()
async def list_object_attributes(
    object_name: str, ctx: Context, pattern: str = "*", include_private: bool = False
) -> dict[str, Any]:
    """List all attributes matching pattern to help LLM discover available methods.

    This tool lists attributes, methods, and properties of an object, helping
    the LLM discover what functionality is available.

    Parameters
    ----------
    object_name : str
        Name of the object to inspect
    pattern : str, optional
        Pattern to match attributes against (supports wildcards)
    include_private : bool, optional
        Whether to include private attributes (starting with _)

    Returns
    -------
    dict
        Dictionary containing:
        - "attributes": list of matching attribute names
        - "methods": list of callable attributes
        - "properties": list of property attributes
        - "total_count": total number of attributes found

    Examples
    --------
    >>> list_object_attributes("str", pattern="*find*")
    {'attributes': ['find', 'rfind'], 'methods': ['find', 'rfind'], ...}

    """
    try:
        session_id = ctx.session_id or "default"
        shell = get_session_shell(session_id)
        if not shell:
            return {"error": "No shell found for session"}
            
        if object_name not in shell.user_ns:
            return {"error": f"Object '{object_name}' not found in namespace"}

        obj = shell.user_ns[object_name]

        all_attrs = dir(obj)

        filtered_attrs = []
        for attr in all_attrs:
            if not include_private and attr.startswith("_"):
                continue
            if fnmatch.fnmatch(attr.lower(), pattern.lower()):
                filtered_attrs.append(attr)

        methods = []
        properties = []
        other_attrs = []

        for attr in filtered_attrs:
            try:
                attr_obj = getattr(obj, attr)
                if callable(attr_obj):
                    methods.append(attr)
                elif isinstance(attr_obj, property):
                    properties.append(attr)
                else:
                    other_attrs.append(attr)
            except:
                other_attrs.append(attr)

        return {
            "attributes": sorted(filtered_attrs),
            "methods": sorted(methods),
            "properties": sorted(properties),
            "other_attributes": sorted(other_attrs),
            "total_count": len(filtered_attrs),
            "pattern_used": pattern,
            "include_private": include_private,
        }
    except Exception as e:
        return {"error": str(e)}


@app.tool()
async def get_docstring(object_name: str, ctx: Context) -> dict[str, Any]:
    """Get just the docstring - lighter than full inspection for understanding APIs.

    This tool provides a lightweight way to get documentation for objects
    without the overhead of full inspection.

    Parameters
    ----------
    object_name : str
        Name of the object to get docstring for

    Returns
    -------
    dict
        Dictionary containing:
        - "docstring": the object's docstring
        - "summary": first line of docstring (brief description)

    Examples
    --------
    >>> get_docstring("print")
    {'docstring': 'print(value, ..., sep=...', 'summary': 'print(value, ..., sep=...)'}

    """
    try:
        session_id = ctx.session_id or "default"
        shell = get_session_shell(session_id)
        if not shell:
            return {"error": "No shell found for session"}
        info = shell.object_inspect(object_name, detail_level=1)

        if not info:
            return {"error": f"Object '{object_name}' not found"}

        docstring = info.get("docstring", "") or ""
        summary = docstring.split("\n")[0] if docstring else ""

        return {
            "docstring": docstring,
            "summary": summary,
            "has_docstring": bool(docstring.strip()),
        }
    except Exception as e:
        return {"error": str(e)}


@app.tool()
async def get_last_exception_info(ctx: Context) -> dict[str, Any]:
    """Get detailed info about last exception to help LLM debug and fix code.

    This tool provides comprehensive information about the most recent exception,
    including the exception type, message, and traceback information.

    Returns
    -------
    dict
        Dictionary containing:
        - "exception_type": type of the exception
        - "exception_message": exception message
        - "traceback": formatted traceback
        - "has_exception": whether there was a recent exception

    Examples
    --------
    >>> get_last_exception_info()
    {'exception_type': 'NameError', 'exception_message': "name 'x' is not defined", ...}

    """
    try:
        session_id = ctx.session_id or "default"
        shell = get_session_shell(session_id)
        if not shell:
            return {"has_exception": False, "error": "No shell found for session"}
        exception_only = shell.get_exception_only()

        if not exception_only or exception_only.strip() == "":
            return {"has_exception": False, "message": "No recent exception found"}

        lines = exception_only.strip().split("\n")
        if lines:
            last_line = lines[-1]
            if ":" in last_line:
                exception_type, exception_message = last_line.split(":", 1)
                exception_type = exception_type.strip()
                exception_message = exception_message.strip()
            else:
                exception_type = last_line.strip()
                exception_message = ""
        else:
            exception_type = "Unknown"
            exception_message = ""

        return {
            "has_exception": True,
            "exception_type": exception_type,
            "exception_message": exception_message,
            "full_exception": exception_only,
            "traceback_lines": lines,
        }
    except Exception as e:
        return {"error": str(e), "has_exception": False}


@app.tool()
async def analyze_syntax_error(code: str) -> dict[str, Any]:
    """Check if code has syntax errors before execution to help LLM validate code.

    This tool performs static analysis of Python code to detect syntax errors
    before execution, helping prevent runtime failures.

    Parameters
    ----------
    code : str
        Python code to analyze for syntax errors

    Returns
    -------
    dict
        Dictionary containing:
        - "valid": whether code has valid syntax
        - "error": error message if invalid
        - "line": line number where error occurs
        - "offset": character offset of error
        - "suggestions": possible fixes (if available)

    Examples
    --------
    >>> analyze_syntax_error("print('hello')")
    {'valid': True}

    >>> analyze_syntax_error("print('hello'")  # missing closing quote
    {'valid': False, 'error': 'EOL while scanning string literal', 'line': 1, ...}

    """
    try:
        compile(code, "<string>", "exec")
        return {"valid": True, "message": "Code has valid syntax"}
    except SyntaxError as e:
        error_info = {
            "valid": False,
            "error": str(e),
            "error_type": "SyntaxError",
            "line": e.lineno,
            "offset": e.offset,
            "text": e.text.strip() if e.text else "",
            "filename": e.filename or "<string>",
        }
        suggestions = []
        error_msg = str(e).lower()

        if "unexpected eof" in error_msg or "eol while scanning" in error_msg:
            suggestions.append("Check for unclosed quotes, parentheses, or brackets")
        elif "invalid syntax" in error_msg:
            suggestions.append("Check for typos in keywords or operators")
        elif "indentation" in error_msg:
            suggestions.append("Check indentation consistency (spaces vs tabs)")
        elif "unmatched" in error_msg:
            suggestions.append("Check for unmatched parentheses or brackets")

        error_info["suggestions"] = suggestions
        return error_info

    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "suggestions": ["Unexpected error during syntax analysis"],
        }


@app.tool()
async def check_code_completeness(code: str, ctx: Context) -> dict[str, Any]:
    """Check if code block is complete to help LLM know when to continue vs execute.

    This tool determines whether a code block is syntactically complete and ready
    for execution, or if it needs additional lines to be valid.

    Parameters
    ----------
    code : str
        Python code to check for completeness

    Returns
    -------
    dict
        Dictionary containing:
        - "status": 'complete', 'incomplete', or 'invalid'
        - "indent": suggested indentation for next line
        - "needs_more": whether more input is needed
        - "reason": explanation of the status

    Examples
    --------
    >>> check_code_completeness("print('hello')")
    {'status': 'complete', 'indent': '', 'needs_more': False}

    >>> check_code_completeness("if True:")
    {'status': 'incomplete', 'indent': '    ', 'needs_more': True}

    """
    try:
        session_id = ctx.session_id or "default"
        shell = get_session_shell(session_id)
        if not shell:
            return {
                "status": "error",
                "indent": "",
                "needs_more": False,
                "reason": "No shell found for session",
                "error": "No shell found for session",
            }
        status, indent = shell.check_complete(code)

        needs_more = status == "incomplete"

        reasons = {
            "complete": "Code block is syntactically complete and ready for execution",
            "incomplete": "Code block needs additional lines to be complete",
            "invalid": "Code contains syntax errors and cannot be completed",
        }

        return {
            "status": status,
            "indent": indent,
            "needs_more": needs_more,
            "reason": reasons.get(status, "Unknown status"),
            "suggested_indent_length": len(indent) if indent else 0,
        }
    except Exception as e:
        return {
            "status": "error",
            "indent": "",
            "needs_more": False,
            "reason": f"Error analyzing code: {str(e)}",
            "error": str(e),
        }


@app.tool()
async def list_available_magics(ctx: Context) -> dict[str, Any]:
    """List all available magic commands to help LLM discover IPython capabilities.

    This tool provides a comprehensive list of available IPython magic commands,
    both line magics (%) and cell magics (%%).

    Returns
    -------
    dict
        Dictionary containing:
        - "line_magics": list of available line magic names
        - "cell_magics": list of available cell magic names
        - "total_line_magics": count of line magics
        - "total_cell_magics": count of cell magics

    Examples
    --------
    >>> list_available_magics()
    {'line_magics': ['cd', 'ls', 'pwd', 'time', ...], 'cell_magics': ['timeit', 'writefile', ...]}

    """
    try:
        session_id = ctx.session_id or "default"
        shell = get_session_shell(session_id)
        if not shell:
            return {"error": "No shell found for session"}
        line_magics = sorted(shell.magics_manager.magics["line"].keys())
        cell_magics = sorted(shell.magics_manager.magics["cell"].keys())

        return {
            "line_magics": line_magics,
            "cell_magics": cell_magics,
            "total_line_magics": len(line_magics),
            "total_cell_magics": len(cell_magics),
            "total_magics": len(line_magics) + len(cell_magics),
        }
    except Exception as e:
        return {"error": str(e)}


@app.tool()
async def get_magic_help(magic_name: str, ctx: Context, magic_type: str = "line") -> dict[str, Any]:
    """Get help for specific magic command to help LLM use magics correctly.

    This tool provides detailed documentation for specific magic commands,
    including usage examples and parameter descriptions.

    Parameters
    ----------
    magic_name : str
        Name of the magic command (without % prefix)
    magic_type : str, optional
        Type of magic: "line" or "cell" (default: "line")

    Returns
    -------
    dict
        Dictionary containing:
        - "help_text": detailed help documentation
        - "exists": whether the magic command exists
        - "magic_type": type of magic (line or cell)
        - "summary": brief description

    Examples
    --------
    >>> get_magic_help("timeit")
    {'help_text': 'Time execution of a Python statement...', 'exists': True, ...}

    """
    try:
        session_id = ctx.session_id or "default"
        shell = get_session_shell(session_id)
        if not shell:
            return {
                "exists": False,
                "error": "No shell found for session",
                "magic_type": magic_type,
                "magic_name": magic_name,
            }
        if magic_type == "line":
            magic_func = shell.find_line_magic(magic_name)
        elif magic_type == "cell":
            magic_func = shell.find_cell_magic(magic_name)
        else:
            magic_func = shell.find_magic(magic_name)

        if not magic_func:
            return {
                "exists": False,
                "error": f"Magic '{magic_name}' not found",
                "magic_type": magic_type,
            }

        help_text = (
            magic_func.__doc__ if magic_func.__doc__ else "No documentation available"
        )
        summary = help_text.split("\n")[0] if help_text else ""

        return {
            "exists": True,
            "help_text": help_text,
            "summary": summary,
            "magic_type": magic_type,
            "magic_name": magic_name,
        }
    except Exception as e:
        return {
            "exists": False,
            "error": str(e),
            "magic_type": magic_type,
            "magic_name": magic_name,
        }


@app.tool()
async def list_dataframes(ctx: Context) -> list[dict]:
    """List all DataFrame variables with their shapes and memory usage.
    
    Returns lightweight metadata about DataFrames in the session.
    Useful for discovering what data is available from previous tool calls.
    
    Returns
    -------
    list[dict]
        Each dict contains: name, type (pandas/polars), rows, columns, memory_mb
    """
    
    session_id = ctx.session_id or "default"
    shell = get_session_shell(session_id)
    if not shell:
        return []
    
    dataframes = []
    for name, obj in shell.user_ns.items():
        if isinstance(obj, (pd.DataFrame, pl.DataFrame)) and not name.startswith('_'):
            df_info = {
                "name": name,
                "type": "pandas" if isinstance(obj, pd.DataFrame) else "polars",
                "rows": obj.shape[0],
                "columns": obj.shape[1]
            }
            if isinstance(obj, pd.DataFrame):
                df_info["memory_mb"] = round(obj.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            else:
                df_info["memory_mb"] = None
            dataframes.append(df_info)
    
    return sorted(dataframes, key=lambda x: x["name"])


@app.tool()
async def session_memory_status(ctx: Context) -> dict:
    """Get current session memory management status.
    
    Shows execution count, DataFrames in memory, and when auto-reset will occur.
    """
    
    session_id = ctx.session_id or "default"
    shell = get_session_shell(session_id)
    if not shell:
        return {
            "error": "No shell found for session",
            "auto_reset_enabled": False,
            "total_executions": 0,
            "executions_since_reset": 0,
            "executions_until_reset": 0,
            "auto_reset_threshold": 0,
            "dataframes_count": 0,
            "dataframes": [],
            "will_reset_on_next": False
        }
    
    dataframes = []
    for name, obj in shell.user_ns.items():
        if isinstance(obj, (pd.DataFrame, pl.DataFrame)) and not name.startswith('_'):
            df_info = {
                "name": name,
                "type": type(obj).__name__,
                "shape": obj.shape
            }
            if isinstance(obj, pd.DataFrame):
                df_info["memory_mb"] = round(obj.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            dataframes.append(df_info)
    
    executions_since_reset = _SMART_MANAGER.execution_counts.get(session_id, 0) - _SMART_MANAGER.last_reset_counts.get(session_id, 0)
    executions_until_reset = _SMART_MANAGER.reset_threshold - executions_since_reset
    
    return {
        "auto_reset_enabled": _SMART_MANAGER.auto_reset_enabled,
        "total_executions": _SMART_MANAGER.execution_counts.get(session_id, 0),
        "executions_since_reset": executions_since_reset,
        "executions_until_reset": max(0, executions_until_reset),
        "auto_reset_threshold": _SMART_MANAGER.reset_threshold,
        "dataframes_count": len(dataframes),
        "dataframes": dataframes[:10],
        "will_reset_on_next": executions_until_reset <= 0 and len(dataframes) > 0 and _SMART_MANAGER.auto_reset_enabled
    }


@app.tool()
async def reset_session_now(ctx: Context) -> dict:
    """Manually trigger a session reset.
    
    This will clear most variables but preserve imports and recent DataFrames.
    """
    session_id = ctx.session_id or "default"
    shell = get_session_shell(session_id)
    if not shell:
        return {
            "status": "Error",
            "message": "No shell found for session"
        }
    _SMART_MANAGER.reset(shell)
    return {
        "status": "Session reset complete",
        "message": "Imports and recent DataFrames preserved"
    }


@app.tool()
async def describe_object(name: str, ctx: Context, sample: int = 5) -> dict:
    """Get structured summary of objects to avoid payload bloat."""
    session_id = ctx.session_id or "default"
    shell = get_session_shell(session_id)
    if not shell:
        return {"error": "No shell found for session"}
    ns = shell.user_ns
    if name not in ns:
        return {"error": f"{name} not in namespace"}

    obj = ns[name]

    if isinstance(obj, pd.DataFrame):
        return {
            "type": "DataFrame",
            "rows": len(obj),
            "columns": list(map(str, obj.columns))[:50],
            "head": obj.head(sample).to_dict("records"),
        }
    
    if isinstance(obj, pl.DataFrame):
        return {
            "type": "DataFrame", 
            "rows": len(obj),
            "columns": list(map(str, obj.columns))[:50],
            "head": obj.head(sample).to_dicts(),
        }

    if isinstance(obj, (np.ndarray, list, tuple, set)):
        return {"type": "array", "len": len(obj), "sample": list(obj)[:sample]}

    if inspect.isfunction(obj) or inspect.isclass(obj):
        return {
            "type": "callable",
            "signature": str(inspect.signature(obj)),
            "doc": (obj.__doc__ or "").split("\n")[0][:500],
        }

    return {"type": type(obj).__name__, "repr": repr(obj)[:500]}
