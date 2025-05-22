from IPython.core.interactiveshell import InteractiveShell
from logai_mcp.session import app, logger
from typing import Optional
import io
import contextlib


_SHELL: InteractiveShell = InteractiveShell.instance()
_SHELL.reset()

async def run_code_in_shell(code: str):
    execution_result = await _SHELL.run_cell_async(code)
    return execution_result.result

@app.tool()
async def execute_python_code(code: str):
    """
    Executes a given string of Python code in the underlying IPython interactive shell.

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
    # Capture stdout and stderr so that users can see print output and error messages
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        result = await run_code_in_shell(code)

    stdout_value = stdout_buffer.getvalue()
    stderr_value = stderr_buffer.getvalue()

    execution_details_dict = {}

    if result is not None:
        execution_details_dict["result"] = result.result

        if result.error_before_exec:
            execution_details_dict["error_before_exec"] = str(result.error_before_exec)

        if result.error_in_exec:
            error_type = type(result.error_in_exec).__name__
            error_msg = str(result.error_in_exec)
            execution_details_dict["error_in_exec"] = f"{error_type}: {error_msg}"


    if stdout_value:
        execution_details_dict["stdout"] = stdout_value.rstrip()

    if stderr_value:
        execution_details_dict["stderr"] = stderr_value.rstrip()

    return execution_details_dict

@app.tool()
async def list_shell_variables() -> list[str]:
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
    system_variables = {'In', 'Out', 'exit', 'quit', 'get_ipython', '_ih', '_oh', '_dh', '_sh', '_ip'}

    if _SHELL.user_ns is None:
        return []

    for name in _SHELL.user_ns.keys():
        if name in system_variables:
            continue
        if name.startswith('_') and name not in {'_', '__', '___', '_i', '_ii', '_iii'} and not name.startswith('_i'):
            continue
        user_vars.append(name)
    return sorted(list(set(user_vars)))

@app.tool()
async def inspect_shell_object(object_name: str, detail_level: int = 0) -> str:
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
    if _SHELL.user_ns is None or object_name not in _SHELL.user_ns:
        return f"Error: Object '{object_name}' not found in the shell namespace."
    try:
        actual_detail_level = min(max(detail_level, 0), 2)
        return _SHELL.object_inspect_text(object_name, detail_level=actual_detail_level)
    except Exception as e:
        return f"Error during inspection of '{object_name}': {str(e)}"

@app.tool()
async def get_shell_history(range_str: str = "", raw: bool = False) -> str:
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
        history_lines = _SHELL.extract_input_lines(range_str=range_str, raw=raw)
        return history_lines
    except Exception as e:
        return f"Error retrieving shell history for range '{range_str}' (raw={raw}): {str(e)}"

@app.tool()
async def run_shell_magic(magic_name: str, line: str, cell: Optional[str] = None):
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
            return _SHELL.run_cell_magic(magic_name, line, cell)
        else:
            return _SHELL.run_line_magic(magic_name, line)
    except Exception as e:
        error_type = type(e).__name__
        return f"Error executing magic command '{magic_name}' (line='{line}', cell present: {cell is not None}): {error_type}: {str(e)}"
