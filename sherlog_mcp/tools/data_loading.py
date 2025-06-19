"""Data-loading related MCP tools extracted from the original monolithic server.

All utilities here *register* with the FastMCP app that lives in
:pyfile:`logai_mcp.session`.
"""

from typing import Any

import pandas as pd
from logai.dataloader.data_loader import DataLoaderConfig, FileDataLoader

from sherlog_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from sherlog_mcp.session import (
    app,
)


def _get_log_file_columns_impl(file_path: str) -> list[str]:
    """Return the column names from the header of a CSV-style log file."""
    cols = pd.read_csv(file_path, nrows=0).columns.tolist()
    return cols


_SHELL.push({"_get_log_file_columns_impl": _get_log_file_columns_impl})


@app.tool()
async def get_log_file_columns(file_path: str, save_as: str):
    """Wrapper that assigns `_get_log_file_columns_impl` result to `save_as` in the shell."""
    code = f"{save_as} = _get_log_file_columns_impl({repr(file_path)})\n{save_as}"
    execution_result = await run_code_in_shell(code)
    return execution_result.result if execution_result else None


get_log_file_columns.__doc__ = _get_log_file_columns_impl.__doc__


def _load_file_log_data_impl(
    file_path: str,
    dimensions: dict,
    log_type: str,
    infer_datetime: bool,
    datetime_format: str,
) -> pd.DataFrame | None:
    """Load structured log data from a local file (e.g., CSV) into a LogAI `LogRecordObject` and return a Pandas DataFrame.

    This tool reads a log file (primarily CSV), processes it using a pandas backend for
    efficiency, and then converts it into a LogAI `LogRecordObject` via an
    intermediate Pandas DataFrame.
    The `LogRecordObject` encapsulates structured log data (timestamps, body, attributes)
    and is the standard input for many downstream LogAI processing tools.
    The LLM is expected to provide the `save_as` key to name this crucial output.

    Parameters
    ----------
    file_path : str
        The path to the log file or a key in `session_vars` whose value is the
        file path. Example: `"./data/server.csv"` or `"input_file_path_var"`.
    dimensions : dict
        A dictionary mapping standard LogAI dimension types (e.g., "timestamp",
        "body", "labels", "span_id") to a list of column names from the input
        file that correspond to that dimension.
        Example: `{"timestamp": ["event_time"], "body": ["message"], "attributes": ["user", "host"]}`.
        It's crucial that column names in this mapping exist in the input file.
        The tool `suggest_dimension_mapping` can help generate a starter mapping.
    log_type : str, default "csv"
        The format of the log file. Currently, "csv" is the primary supported type,
        but LogAI's `FileDataLoader` might support others.
    infer_datetime : bool, default False
        If True, LogAI will attempt to automatically infer the format of datetime
        strings in the timestamp column(s). If False (default), `datetime_format`
        must be provided and will be used for parsing.
    datetime_format : str, default "%Y-%m-%dT%H:%M:%SZ"
        The `strftime` format string used to parse datetime strings if
        `infer_datetime` is False. The default is ISO 8601 format.
        Example: `"%d/%b/%Y:%H:%M:%S %z"` for Apache-like timestamps.
    save_as : str
        The **required** key under which the loaded `LogRecordObject` as a Pandas DataFrame will be stored in
        `session_vars`. This key must be provided by the caller (LLM).

    Returns
    -------
    pd.DataFrame
        An object containing the loaded and structured log data, including
        timestamps, loglines (body), and attributes.

    Side Effects
    ------------
    - Raises a `ValueError` if any column specified in the `dimensions`
      mapping is not found in the header of the input file.

    Examples
    --------
    >>> dims = {"timestamp": ["tstamp"], "body": ["msg"]}
    >>> load_file_log_data(file_path="data.csv", dimensions=dims, save_as="loaded_logs")

    See Also
    --------
    get_log_file_columns : To get column names to help create the `dimensions` mapping.
    suggest_dimension_mapping : To automatically generate a `dimensions` mapping.
    preprocess_log_data : For cleaning and transforming the loaded log data.
    peek_file : For a quick preview of the file content.

    Notes
    -----
    - The `dimensions` mapping is critical for correct parsing. Ensure that all
      columns listed in `dimensions` actually exist in the input file.

    """
    config = DataLoaderConfig()
    config.filepath = file_path
    config.dimensions = dimensions
    config.log_type = log_type
    config.infer_datetime = infer_datetime
    config.datetime_format = datetime_format
    file_data_loader = FileDataLoader(config)

    return file_data_loader.load_data().to_dataframe()


_SHELL.push({"_load_file_log_data_impl": _load_file_log_data_impl})


@app.tool()
async def load_file_log_data(
    file_path: Any,
    dimensions: dict,
    log_type: str = "csv",
    infer_datetime: bool = False,
    datetime_format: str = "%Y-%m-%dT%H:%M:%SZ",
    *,
    save_as: str,
) -> pd.DataFrame | None:
    """Wrapper that assigns `_load_file_log_data_impl` result to `save_as` in the shell."""
    code = f"{save_as} = _load_file_log_data_impl({repr(file_path)}, {repr(dimensions)}, {repr(log_type)}, {repr(infer_datetime)}, {repr(datetime_format)})\n{save_as}"
    execution_result = await run_code_in_shell(code)
    return execution_result.result if execution_result else None


def _default_dimension_mapping(cols: list[str]) -> pd.DataFrame:
    def _norm(name: str) -> str:
        return name.lower().replace(" ", "").replace("_", "").replace("-", "")

    ts_candidates = [
        c for c in cols if _norm(c) in {"timestamp", "time", "date", "datetime"}
    ]
    body_candidates_exact = {"body", "message", "url", "log", "text"}
    body_candidates = [c for c in cols if _norm(c) in body_candidates_exact]

    label_keywords = {
        "label",
        "labels",
        "status",
        "statuscode",
        "responsestatus",
        "responsecode",
        "severity",
        "level",
        "class",
        "outcome",
        "anomaly",
    }
    label_candidates = [c for c in cols if any(kw in _norm(c) for kw in label_keywords)]

    res = {
        "timestamp": ts_candidates[:1],
        "body": body_candidates[:1] if body_candidates else [],
        "span_id": [],
        "labels": label_candidates[:1],
    }

    return pd.DataFrame(res)


_SHELL.push({"_default_dimension_mapping": _default_dimension_mapping})


def _suggest_dimension_mapping_impl(file_path: str) -> pd.DataFrame:
    cols = pd.read_csv(file_path, nrows=0).columns.tolist()
    return _default_dimension_mapping(cols)


_SHELL.push({"_suggest_dimension_mapping_impl": _suggest_dimension_mapping_impl})


@app.tool()
async def suggest_dimension_mapping(
    file_path: str, save_as: str
) -> pd.DataFrame | None:
    """Generate a heuristic ``dimensions`` mapping for a log file, saved to session_vars.

    Inspects CSV column names and suggests a mapping for "timestamp", "body",
    "labels", etc., useful for `load_file_log_data`.

    The heuristics look for common keywords in column names (e.g., "time",
    "date" for timestamp; "message", "log" for body; "level", "status" for labels).

    Parameters
    ----------
    file_path : str
        The path to the CSV log file for which to suggest a dimension mapping.
        This argument is **not** resolved from `session_vars` directly by this
        function's signature but is passed to `get_log_file_columns` which *does*
        resolve it if it's a key. For clarity, providing a direct path string is typical.
    save_as : str
        The **required** key under which the suggested dimension mapping (a dictionary)
        will be stored in a dataframe with the name `save_as`. This key must be provided by the caller (LLM).

    Returns
    -------
    dict
        A dictionary representing the suggested dimension mapping.
        Example: `{"timestamp": ["event_date"], "body": ["log_message"], "labels": ["severity_level"], "span_id": []}`.
        If no suitable columns are found for a dimension, its list will be empty.

    Side Effects
    ------------
    - Stores the suggested mapping dictionary in `session_vars` under the key
      specified by `save_as` or an auto-generated key.
    - Internally calls `get_log_file_columns` which also has side effects on
      `session_vars` if its `save_as` is not managed carefully (though this
      tool calls `get_log_file_columns` without specifying `save_as`, so
      `get_log_file_columns` will use its own auto-naming for its result).

    Examples
    --------
    # File 'server.csv' has columns: 'request_time', 'url_path', 'status_code'
    >>> suggest_dimension_mapping(file_path="server.csv", save_as="suggested_dims")
    # Returns: {'timestamp': ['request_time'], 'body': ['url_path'], 'span_id': [], 'labels': ['status_code']} (example)
    # session_vars["suggested_dims"] will store this dictionary.

    See Also
    --------
    load_file_log_data : Uses the output of this tool for its `dimensions` parameter.
    get_log_file_columns : Called internally to fetch column names.

    Notes
    -----
    - The quality of the suggestion depends on how conventionally the columns
      are named. Manual review and adjustment of the suggested mapping are often necessary.
    - This tool currently primarily targets CSV files, as it uses
      `get_log_file_columns` which is CSV-oriented.

    """
    code = f"{save_as} = _suggest_dimension_mapping_impl({repr(file_path)})\n{save_as}"
    execution_result = await run_code_in_shell(code)
    return execution_result.result if execution_result else None
