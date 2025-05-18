"""Central session & utility helpers for LogAI FastMCP server.

This module owns the FastMCP *app* instance plus the in-memory scratch-pad
(`session_vars`) that lets individual tool calls communicate with each other.
All other modules should *only* import what they need from here instead of
instantiating additional `FastMCP` objects.
"""

from typing import Any, Dict, List
from datetime import datetime
import functools
import logging

import nltk
import nltk.downloader
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# FastMCP application singleton
# ---------------------------------------------------------------------------

app = FastMCP(name="LogAIMCP")

# ---------------------------------------------------------------------------
# Download NLTK resources (best-effort)
# ---------------------------------------------------------------------------

for _resource in [
    "tokenizers/punkt",
    "corpora/wordnet",
    "taggers/averaged_perceptron_tagger",
]:
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_resource.split("/")[-1], quiet=True)

# ---------------------------------------------------------------------------
# Session state & helper utilities
# ---------------------------------------------------------------------------

session_vars: Dict[str, Any] = {}
"""Stores objects produced by tool calls so that later calls can reference
    them by *name* instead of passing bulky objects around."""

session_meta: Dict[str, Dict[str, Any]] = {}
"""Provenance information for every key in :data:`session_vars`.  Set by
    :func:`log_tool`."""


def _resolve(arg: Any) -> Any:
    """Return the stored object if *arg* is a key in :data:`session_vars`."""
    return session_vars.get(arg, arg) if isinstance(arg, str) else arg


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger("LogAIMCP")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _short_repr(obj: Any, max_len: int = 200) -> str:
    """Return a safe, truncated ``repr`` for logging purposes."""
    try:
        r = repr(obj)
        if len(r) > max_len:
            r = r[:max_len] + "..."
        return r
    except Exception:
        return f"<{type(obj).__name__}>"


# ---------------------------------------------------------------------------
# Decorator to register & log tool calls
# ---------------------------------------------------------------------------

def log_tool(func):
    """Decorator that logs entry/exit of a FastMCP *tool* and records provenance.

    The wrapped callable is assumed to be decorated with ``@app.tool()`` higher
    up the stack.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info("→ %s args=%s kwargs=%s", func.__name__, _short_repr(args), _short_repr(kwargs))

        before_keys = set(session_vars.keys())
        # Identify keys that are explicitly named for saving via kwargs
        explicit_save_keys_from_kwargs = set()
        for kw_name, kw_value in kwargs.items():
            if kw_name.startswith("save") and kw_value is not None and isinstance(kw_value, str):
                explicit_save_keys_from_kwargs.add(kw_value)

        try:
            result = func(*args, **kwargs) # func might modify session_vars

            # Keys that were newly created during this tool call
            newly_created_keys = set(session_vars.keys()) - before_keys

            # Keys that were explicitly named in kwargs for saving and were overwritten
            # These are keys that were in kwargs, existed before, and still exist now.
            overwritten_explicit_keys = explicit_save_keys_from_kwargs.intersection(before_keys).intersection(session_vars.keys())

            # All keys for which metadata should be set or updated by this tool call
            keys_to_update_meta_for = newly_created_keys.union(overwritten_explicit_keys)

            kw_filtered = {k: v for k, v in kwargs.items() if not k.startswith("save")}
            for name_to_meta in keys_to_update_meta_for:
                # Ensure the key actually exists in session_vars (it should be, but as a safeguard)
                if name_to_meta in session_vars:
                    session_meta[name_to_meta] = {
                        "source_tool": func.__name__,
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "args_repr": _short_repr(args),
                        "kwargs_repr": _short_repr(kw_filtered),
                    }

                    # New: Save the actual filtered kwargs dictionary to session_vars
                    params_key = f"params_for_{name_to_meta}"
                    session_vars[params_key] = kw_filtered  # Store the actual dictionary
                    logger.info(f"Saved input keyword arguments for '{name_to_meta}' to session_vars['{params_key}'].")

                    # New: Add metadata for this new params_key entry in session_meta
                    session_meta[params_key] = {
                        "source_tool": func.__name__, # Attributed to the same tool that produced the result
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                        "description": f"Input keyword arguments (excluding 'save_as'-style args) for the tool call that produced/updated session_vars['{name_to_meta}']",
                        "content_type": "tool_parameters", # Custom field to identify this as a params dict
                        "original_result_key": name_to_meta # Link back to the result
                    }

            logger.info("← %s result=%s", func.__name__, _short_repr(result))
            return result
        except Exception as exc:
            logger.exception("! %s raised %s", func.__name__, exc)
            raise

    return wrapper


# ---------------------------------------------------------------------------
# Generic / REPL-style utilities (registered as tools right here)
# ---------------------------------------------------------------------------


@app.tool()
@log_tool
def list_session_vars(show_values: bool = False, include_meta: bool = False):
    """Inspect what objects are currently stored in the session's shared memory.

    This tool allows for introspection of `session_vars`, a dictionary acting
    as a scratch-pad for data passed between different tool calls. It can
    display variable names, their values, and associated metadata.

    Parameters
    ----------
    show_values : bool, default False
        If *False* (default), only the names of stored variables are returned
        (or metadata if `include_meta` is True).
        If *True*, the actual values of the variables are included in the output.
    include_meta : bool, default False
        When *True*, provenance information for each variable (e.g., which tool
        created it, when, and with what arguments) is retrieved from
        `session_meta` and included in the output.

    Returns
    -------
    list[str] | dict[str, Any]
        The content and structure of the return value depend on the flags:
        - `show_values=False, include_meta=False` (default):
          `list[str]` containing only the names of variables in `session_vars`.
          If `session_vars` is empty, an empty list `[]` is returned, definitively
          indicating that no variables are currently stored in the session.
          Example: `["df_loaded", "user_settings"]`
        - `show_values=True, include_meta=False`:
          `dict[str, Any]` mapping variable names to their stored values.
          Example: `{"df_loaded": pd.DataFrame(...), "user_settings": {"theme": "dark"}}`
        - `show_values=False, include_meta=True`:
          `dict[str, dict]` mapping variable names to their metadata.
          Example: `{"df_loaded": {"source_tool": "load_data", ...}}`
        - `show_values=True, include_meta=True`:
          `dict[str, dict]` mapping variable names to a dictionary containing
          both their "value" and "meta" information.
          Example: `{"df_loaded": {"value": pd.DataFrame(...), "meta": {"source_tool": "load_data", ...}}}`

    See Also
    --------
    inspect_session_variable : For a detailed view of a single variable.
    py : To store new variables or modify existing ones in `session_vars`.

    Notes
    -----
    - `session_vars` holds the actual data objects.
    - `session_meta` stores provenance details for items in `session_vars`.
    - This tool is read-only and does not modify the session state.
    - If `session_vars` is empty, this tool will return an empty collection
      (e.g., `[]` or `{}`) appropriate to the specified parameters,
      signifying that no variables are currently stored. Calling the tool
      repeatedly will not yield a different result unless another tool
      modifies `session_vars`.
    """
    if include_meta:
        if show_values:
            return {
                name: {"value": val, "meta": session_meta.get(name)}
                for name, val in session_vars.items()
            }
        return {name: session_meta.get(name) for name in session_vars.keys()}

    return session_vars if show_values else list(session_vars.keys())


@app.tool()
@log_tool
def py(expr: str, *, save_as: str | None = None):
    """Evaluate arbitrary Python code within the MCP session, similar to a REPL.

    This tool provides a powerful way to execute Python expressions or statements
    dynamically. It can be used for calculations, data manipulation, or interacting
    with objects stored in `session_vars`.

    The execution environment provides access to `session_vars` via a shorthand
    variable `v`. For example, to access a DataFrame stored as "my_df" in
    `session_vars`, you could use `v["my_df"]` in your expression.

    Parameters
    ----------
    expr : str
        A string containing the Python code to be evaluated.
        - If `expr` is an expression (e.g., "1 + 1", "v['my_data'].shape"),
          the result of the expression is returned and stored.
        - If `expr` is a statement (e.g., "x = 10; y = x * 2",
          "v['new_list'] = [1,2,3]"), the tool attempts to capture a
          result if a variable `_` is assigned within the exec scope,
          otherwise `None` might be stored. For direct assignment to
          `session_vars`, use `v['key'] = value` within the expression.
    save_as : str | None, default None
        The key under which the result of the evaluation will be stored in
        `session_vars`.
        If `None` (default), a unique name is automatically generated using
        the pattern "py_output_N" (e.g., "py_output_1", "py_output_2", ...),
        incrementing N until an unused key is found. A log message will
        indicate the auto-generated name.
        If a name is provided, it will overwrite any existing variable with
        the same name in `session_vars`.

    Returns
    -------
    Any
        The result of the evaluated Python expression.
        If the expression is a statement, it might return the value assigned to
        a special variable `_` within the executed code, or `None`.
        If an error occurs during evaluation (e.g., `SyntaxError`, `NameError`),
        a string describing the error is returned (e.g.,
        "[PY-ERROR] NameError: name 'undefined_var' is not defined").

    Side Effects
    ------------
    - Stores the evaluation result (or error string) in `session_vars` under
      the key specified by `save_as` or an auto-generated key.
    - The executed code can modify `session_vars` directly if it uses the `v`
      object (e.g., `expr="v['existing_var'] = v['existing_var'] * 2"`).

    Examples
    --------
    >>> py(expr="10 + 5 * 2", save_as="calculation_result")
    # session_vars["calculation_result"] will be 20

    >>> py(expr="v['loaded_data'].head(2)", save_as="data_preview")
    # Assumes "loaded_data" exists in session_vars.
    # session_vars["data_preview"] will contain the first 2 rows.

    >>> py(expr="import numpy as np; arr = np.array([1,2,3]); _ = arr.sum()")
    # session_vars["py_output_1"] (or similar) will be 6

    >>> py(expr="v['my_list'] = ['a', 'b', 'c']")
    # session_vars["py_output_1"] (or similar) will be None, but
    # session_vars["my_list"] will be ['a', 'b', 'c']

    See Also
    --------
    list_session_vars : To see what's available in `session_vars`.
    inspect_session_variable : To examine a specific variable in `session_vars`.

    Notes
    -----
    - Exercise caution with arbitrary code execution, especially if the source
      of `expr` is untrusted.
    - The `eval()` and `exec()` Python built-ins are used internally.
    - For multi-line statements, use semicolons or ensure the string `expr`
      is a valid multi-line Python string.
    """
    v = session_vars  # Give users a shorthand to reference the dict
    try:
        code_obj = compile(expr, "<py-eval>", "eval")
        result = eval(code_obj, {}, {"v": v})
    except SyntaxError:
        local_env = {"v": v}
        try:
            code_obj = compile(expr, "<py-exec>", "exec")
            exec(code_obj, {}, local_env)
            result = local_env.get("_", None)
        except Exception as exc:
            logger.warning("py tool execution failed: %s: %s", type(exc).__name__, exc)
            result = f"[PY-ERROR] {type(exc).__name__}: {exc}"
    except Exception as exc:
        logger.warning("py tool execution failed: %s: %s", type(exc).__name__, exc)
        result = f"[PY-ERROR] {type(exc).__name__}: {exc}"

    _save_as_key = save_as
    if _save_as_key is None:
        base_name = "py_output"
        counter = 1
        _save_as_key = f"{base_name}_{counter}"
        while _save_as_key in session_vars:
            counter += 1
            _save_as_key = f"{base_name}_{counter}"
        logger.info(f"\'save_as\' not provided for py tool. Saving result to session_vars as \'{_save_as_key}\'.")

    session_vars[_save_as_key] = result
    return result


@app.tool()
@log_tool
def inspect_session_variable(
    variable_name: str,
    preview_elements: int = 5,
    max_preview_str_len: int = 500,
):
    """Return rich inspection information for a specified variable in `session_vars`.

    This tool provides a detailed look at a single object stored in the
    session's shared memory (`session_vars`). It shows the variable's name,
    type, existence, and, for common data structures like pandas DataFrames,
    Series, or NumPy arrays, it includes shape, dtypes, and a data preview.

    Parameters
    ----------
    variable_name : str
        The exact name (key) of the variable within `session_vars` that
        needs to be inspected. This argument is **not** resolved from
        `session_vars` itself; it must be a string literal representing the key.
    preview_elements : int, default 5
        For data structures that support it (like pandas DataFrames/Series or
        1D/2D NumPy arrays), this determines how many initial elements or rows
        are shown in the preview. For 2D NumPy arrays, it previews a slice
        `[:preview_elements, :preview_elements]`.
    max_preview_str_len : int, default 500
        For objects where a detailed preview isn't available (e.g., custom
        objects, complex dicts), this limits the length of their string
        representation (`repr()`) in the preview.

    Returns
    -------
    dict[str, Any]
        A dictionary containing various details about the inspected variable:
        - "name" (str): The `variable_name` that was requested.
        - "type" (str): The string representation of the variable's type
          (e.g., "<class 'pandas.core.frame.DataFrame'>"). "N/A" if not found.
        - "exists" (bool): True if `variable_name` was found in `session_vars`,
          False otherwise.
        - "length" (int | None): The result of `len(obj)` if applicable and
          successful, otherwise None.
        - "shape" (list[int] | None): For pandas DataFrames, Series, and NumPy
          arrays, this is a list representing their dimensions (e.g., `[100, 5]`
          for a DataFrame with 100 rows and 5 columns). None otherwise.
        - "dtypes" (dict[str, str] | str | None):
            - For pandas DataFrames: a dictionary mapping column names to their
              data types (as strings).
            - For pandas Series or NumPy arrays: a string representing the data type.
            - None otherwise.
        - "preview" (str): A string representation of a preview of the object's
          content. This varies:
            - pandas DataFrame/Series: `head(preview_elements).to_string()`.
            - NumPy ndarray: `np.array_str()` of the initial slice.
            - Other types: `repr(obj)`, truncated by `max_preview_str_len`.
          "N/A" if not found.
        - "error" (str | None): An error message if the variable was not found
          or if an exception occurred during inspection. None if successful.

    Examples
    --------
    # Assuming "my_dataframe" is a pandas DataFrame in session_vars
    >>> inspect_session_variable(variable_name="my_dataframe")
    {
        "name": "my_dataframe",
        "type": "<class 'pandas.core.frame.DataFrame'>",
        "exists": True,
        "length": 100, # Example value
        "shape": [100, 5], # Example value
        "dtypes": {"col1": "int64", "col2": "object"}, # Example
        "preview": "   col1 col2\\n0   1   a\\n1   2   b...", # Example
        "error": None
    }

    # If "non_existent_var" is not in session_vars
    >>> inspect_session_variable(variable_name="non_existent_var")
    {
        "name": "non_existent_var",
        "type": "N/A",
        "exists": False,
        # ... other fields would be None or default ...
        "preview": "N/A",
        "error": "Variable 'non_existent_var' not found."
    }

    See Also
    --------
    list_session_vars : To get a list of all available variables in `session_vars`.
    peek_file : For a quick look at file contents before loading.

    Notes
    -----
    - This tool is read-only concerning `session_vars`.
    - It relies on `pandas` and `numpy` for detailed previews of their
      respective objects. If these libraries are not available in the
      environment where this code runs (unlikely for LogAI), previews
      might be more basic.
    """
    import pandas as _pd
    import numpy as _np

    details: Dict[str, Any] = {
        "name": variable_name,
        "type": "N/A",
        "exists": False,
        "length": None,
        "shape": None,
        "dtypes": None,
        "preview": "N/A",
        "error": None,
    }

    if variable_name not in session_vars:
        details["error"] = f"Variable '{variable_name}' not found."
        logger.warning(details["error"])
        return details

    obj = session_vars[variable_name]
    details.update({"exists": True, "type": str(type(obj))})

    try:
        if hasattr(obj, "__len__"):
            try:
                details["length"] = len(obj)
            except TypeError:
                pass

        if isinstance(obj, _pd.DataFrame):
            details["shape"] = list(obj.shape)
            details["dtypes"] = {c: str(t) for c, t in obj.dtypes.items()}
            details["preview"] = obj.head(preview_elements).to_string(max_rows=preview_elements)
        elif isinstance(obj, _pd.Series):
            details["shape"] = list(obj.shape)
            details["dtypes"] = str(obj.dtype)
            details["preview"] = obj.head(preview_elements).to_string(max_rows=preview_elements)
        elif isinstance(obj, _np.ndarray):
            details["shape"] = list(obj.shape)
            details["dtypes"] = str(obj.dtype)
            slice_ = obj[:preview_elements] if obj.ndim == 1 else obj[:preview_elements, :preview_elements]
            details["preview"] = _np.array_str(slice_, max_line_width=80)
        else:
            preview = _short_repr(obj, max_preview_str_len)
            details["preview"] = preview
    except Exception as exc:
        logger.exception("Error inspecting variable %s: %s", variable_name, exc)
        details["error"] = f"{type(exc).__name__}: {exc}"

    return details


@app.tool()
@log_tool
def peek_file(file_path: str, n_lines: int = 10) -> List[str]:
    """Return the first `n_lines` of a text file without fully loading or parsing it.

    This tool is useful for a quick inspection of a file's content, for example,
    to understand its structure, check for headers, or get a glimpse of the data
    before deciding how to process it with other tools (e.g., `load_file_log_data`).

    Parameters
    ----------
    file_path : str
        The absolute or relative path to the text file that needs to be peeked into.
        This argument is **not** resolved from `session_vars`; it must be a
        string literal representing the file path.
    n_lines : int, default 10
        The number of lines to read from the beginning of the file.
        If the file has fewer than `n_lines`, all lines in the file are returned.

    Returns
    -------
    List[str]
        A list of strings, where each string is a line read from the file,
        with trailing newline characters (e.g., '\\n', '\\r\\n') removed.
        If the file cannot be opened or read (e.g., file not found, permission
        denied), an exception will be raised by the underlying file operation.

    Examples
    --------
    # Peek at the first 5 lines of 'data.csv'
    >>> peek_file(file_path="data.csv", n_lines=5)
    ['header1,header2,header3', 'val1,val2,val3', ...] # Example output

    # Peek at a file with fewer than n_lines
    >>> peek_file(file_path="short_file.txt", n_lines=20)
    # If short_file.txt has only 3 lines, it returns those 3 lines.

    See Also
    --------
    load_file_log_data : For loading and parsing structured log data from files.
    get_log_file_columns : To get column names from a CSV file.

    Notes
    -----
    - The file is opened in read mode ('r') with UTF-8 encoding. Errors during
      decoding are replaced with a replacement character.
    - This tool does not store any data in `session_vars`.
    - It's designed for text files. Peeking into binary files might produce
      unreadable output.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        return [next(fh).rstrip("\\n") for _ in range(n_lines)] 