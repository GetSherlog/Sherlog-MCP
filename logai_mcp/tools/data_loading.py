"""Data-loading related MCP tools extracted from the original monolithic server.

All utilities here *register* with the FastMCP app that lives in
:pyfile:`logai_mcp.session`.
"""

from typing import Any, Dict, List, Union, Literal
import glob
import os

import pandas as pd
import polars as pl
from datetime import datetime

from logai_mcp.dataframe_utils import convert_to_logai_input
from logai_mcp.session import (
    app,
    log_tool,
    _resolve,
    session_vars,
    session_meta,
    logger,
)

# --- LogAI imports -----------------------------------------------------------
# from logai.dataloader.data_loader import DataLoaderConfig, FileDataLoader # Replaced by Polars + custom conversion
from logai.dataloader.openset_data_loader import (
    OpenSetDataLoader,
    OpenSetDataLoaderConfig,
)
from logai.utils import constants

__all__ = [
    "get_log_file_columns",
    "load_file_log_data",
    "load_custom_parsed_csv_log",
    "suggest_dimension_mapping",
    "combine_csv_files",
    "load_openset_log_data",
]

# ---------------------------------------------------------------------------
# Basic CSV helpers
# ---------------------------------------------------------------------------


@app.tool()
@log_tool
def get_log_file_columns(file_path: str, *, save_as: str) -> List[str]:
    """Return the column names from the header of a CSV-style log file.

    This tool reads the first line of the specified CSV file to extract its
    column headers. The result (a list of column names) is returned and
    also stored in `session_vars` under the explicitly provided `save_as` key.
    This facilitates direct use of the named output by the LLM for subsequent analysis
    or tool calls.

    Parameters
    ----------
    file_path : str
        The path to the CSV file. This can be an absolute path or a path
        relative to the workspace. This argument can also be a string key
        to retrieve the actual file path from `session_vars` (resolved via `_resolve`).
    save_as : str
        The **required** key under which the list of column names will be stored in
        `session_vars`. This key must be provided by the caller (LLM).

    Returns
    -------
    List[str]
        A list of strings, where each string is a column name from the
        CSV file's header.

    Side Effects
    ------------
    - Stores the list of column names in `session_vars` under the key specified
      by `save_as` or an auto-generated key.

    Examples
    --------
    # Assuming 'logs/data.csv' contains:
    # timestamp,message,user_id
    # ...data...
    >>> get_log_file_columns(file_path="logs/data.csv", save_as="csv_headers")
    # Returns: ['timestamp', 'message', 'user_id']
    # session_vars["csv_headers"] will be ['timestamp', 'message', 'user_id']

    # Using an auto-generated name
    >>> get_log_file_columns(file_path="auth.log.csv")
    # Returns: ['event_time', 'source_ip', 'status'] (example)
    # session_vars["auth_log_columns_1"] (or similar) will store the list.

    See Also
    --------
    load_file_log_data : For loading the entire content of the log file.
    suggest_dimension_mapping : To get a suggested mapping based on column names.
    peek_file : For a quick look at the first few lines of any file.

    Notes
    -----
    - Relies on `pandas.read_csv()` to infer columns, so it expects a
      standard CSV format with a header row.
    - If `file_path` is a key in `session_vars`, its corresponding value (the
      actual path) will be used.
    """
    file_path = _resolve(file_path)
    cols = pd.read_csv(file_path).columns.tolist()
    session_vars[save_as] = cols
    logger.info(f"Saved log file columns to session_vars as '{save_as}'.")
    return cols


# ---------------------------------------------------------------------------
# General-purpose file loader using LogAI's FileDataLoader
# ---------------------------------------------------------------------------


@app.tool()
@log_tool

def load_file_log_data(
    file_path: Any,
    dimensions: dict,
    log_type: str = "csv",
    infer_datetime: bool = False,
    datetime_format: str = "%Y-%m-%dT%H:%M:%SZ",
    *,
    save_as: str,
) -> Any:
    """Load structured log data from a local file (e.g., CSV) into a LogAI `LogRecordObject`.

    This tool reads a log file (primarily CSV), processes it using a Polars backend for
    efficiency, and then converts it into a LogAI `LogRecordObject` via an
    intermediate Pandas DataFrame. This `LogRecordObject` is then stored in
    `session_vars` under the explicitly provided `save_as` key.
    The `LogRecordObject` encapsulates structured log data (timestamps, body, attributes)
    and is the standard input for many downstream LogAI processing tools.
    The LLM is expected to provide the `save_as` key to name this crucial output.

    Parameters
    ----------
    file_path : Any
        The path to the log file or a key in `session_vars` whose value is the
        file path. This argument is resolved via `_resolve`.
        Example: `"./data/server.csv"` or `"input_file_path_var"`.
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
        The **required** key under which the loaded `LogRecordObject` will be stored in
        `session_vars`. This key must be provided by the caller (LLM).

    Returns
    -------
    logai.dataloader.data_structures.LogRecordObject
        An object containing the loaded and structured log data, including
        timestamps, loglines (body), and attributes.

    Side Effects
    ------------
    - Stores the `LogRecordObject` in `session_vars` under the key specified by
      `save_as` or an auto-generated key.
    - Raises a `ValueError` if any column specified in the `dimensions`
      mapping is not found in the header of the input file.

    Examples
    --------
    >>> dims = {"timestamp": ["tstamp"], "body": ["msg"]}
    >>> load_file_log_data(file_path="data.csv", dimensions=dims, save_as="loaded_logs")
    # session_vars["loaded_logs"] will contain the LogRecordObject.

    See Also
    --------
    get_log_file_columns : To get column names to help create the `dimensions` mapping.
    suggest_dimension_mapping : To automatically generate a `dimensions` mapping.
    preprocess_log_data : For cleaning and transforming the loaded log data.
    peek_file : For a quick preview of the file content.

    Notes
    -----
    - The `file_path` argument is resolved using `_resolve`, meaning it can be
      a direct path string or a key in `session_vars`.
    - The `dimensions` mapping is critical for correct parsing. Ensure that all
      columns listed in `dimensions` actually exist in the input file.
    """
    file_path = _resolve(file_path)

    if log_type.lower() != "csv":
        logger.warning(
            f"load_file_log_data with Polars backend currently primarily supports CSV. "
            f"log_type='{log_type}' might not work as expected. Attempting to read as CSV."
        )
    
    try:
        # Attempt to read header to check file existence and basic format
        # Use ignore_errors=True for header check robustness, separator for explicit CSV.
        header_df = pl.read_csv(file_path, n_rows=0, separator=',', ignore_errors=True)
        header_cols = header_df.columns
        
        if not header_cols and os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > 0:
                 logger.warning(f"Polars couldn't parse headers for {file_path} (size: {file_size} bytes). It might be empty, not a standard CSV, or have encoding issues.")
            # If file exists but header_cols is empty, it might be an empty file or unparsable.
            # Subsequent full read might fail more informatively or succeed if it's an edge case.
        elif not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

    except Exception as e: # Catch broad exceptions during initial header read
        logger.error(f"Failed to read header from {file_path} using Polars: {e}")
        raise ValueError(f"Could not read header from file {file_path}. Ensure it's a valid CSV-like file. Error: {e}")

    # Validate dimensions against header_cols
    # This check should happen after successfully getting header_cols (even if empty list for empty file)
    if header_cols: 
        missing_cols = {
            dim_col
            for dim_list in dimensions.values() # dimensions is the log_record_meta
            for dim_col in dim_list
            if dim_col not in header_cols
        }
        if missing_cols:
            raise ValueError(
                "Dimension mapping refers to columns that do not exist in the file: "
                f"{sorted(missing_cols)} | Available columns in file: {list(header_cols)}"
            )
    else: # If header_cols is empty (e.g. empty file or unparsable header)
        any_expected_cols = any(dim_list for dim_list in dimensions.values() if dim_list)
        if any_expected_cols:
             raise ValueError(
                f"Dimension mapping expects columns, but no headers found or parsed in {file_path} (file might be empty or not a valid CSV)."
            )

    # New data loading with Polars + convert_to_logai_input
    try:
        if infer_datetime:
            # try_parse_dates can be slow; only use if requested.
            # ignore_errors helps skip rows that can't be parsed, rather than failing the whole read.
            raw_df = pl.read_csv(file_path, try_parse_dates=True, ignore_errors=True)
        else:
            # Read with specific datetime parsing if format is given.
            raw_df = pl.read_csv(file_path, try_parse_dates=False, ignore_errors=True)
            # `dimensions` keys like "timestamp" map to LogAI's internal constants.
            timestamp_cols_to_parse = dimensions.get("timestamp", []) 
            
            if timestamp_cols_to_parse:
                parse_expressions = []
                for col_name in timestamp_cols_to_parse:
                    if col_name in raw_df.columns:
                        parse_expressions.append(
                            pl.col(col_name)
                            .cast(pl.String, strict=False) # Cast to string, non-strict
                            .str.strptime(pl.Datetime, format=datetime_format, strict=False, exact=True) 
                            .alias(col_name)
                        )
                if parse_expressions:
                    raw_df = raw_df.with_columns(parse_expressions)
        
        if raw_df.height == 0 and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            logger.warning(f"Polars read 0 rows from {file_path} though file is not empty. Check CSV format, content, and encoding.")

        # The `dimensions` dict serves as `log_record_meta`
        result = convert_to_logai_input(polars_df=raw_df, log_record_meta=dimensions)

    except pl.PolarsError as pe: # Catch specific Polars errors
        logger.error(f"Polars specific error during CSV read for {file_path}: {pe}")
        raise RuntimeError(f"Failed to process data from {file_path} due to a Polars error: {pe}")
    except Exception as e: # Catch other errors during conversion or general issues
        logger.error(f"Error during Polars CSV read or conversion to LogRecordObject for {file_path}: {e}")
        raise RuntimeError(f"Failed to load or convert data from {file_path} using Polars backend: {e}")

    _save_as_key = save_as
    if _save_as_key is None:
        # Ensure file_path is a string for os.path operations
        str_file_path = str(file_path)
        base_name = f"{os.path.splitext(os.path.basename(str_file_path))[0]}_log_record"
        counter = 1
        _save_as_key = f"{base_name}_{counter}"
        while _save_as_key in session_vars:
            counter += 1
            _save_as_key = f"{base_name}_{counter}"
        logger.info(f"'save_as' not provided for load_file_log_data. Saving result to session_vars as '{_save_as_key}'.")
    session_vars[save_as] = result
    logger.info(f"Saved loaded LogRecordObject to session_vars as '{save_as}'.")
    return result


# ---------------------------------------------------------------------------
# Custom CSV Loader with Robust Parsing and Detailed Errors
# ---------------------------------------------------------------------------

@app.tool()
@log_tool
def load_custom_parsed_csv_log(
    file_path: str,
    expected_datetime_format: str = "%d/%b/%Y:%H:%M:%S",
    custom_header: List[str] | None = None,
    url_column_indices: List[int] | None = None,
    timestamp_column_index: int = 1,
    status_column_is_last: bool = True,
    *,
    save_as: str,
) -> pd.DataFrame:
    """Load and parse a CSV log file with custom logic, returning a Pandas DataFrame.

    This tool is designed for CSV files where standard parsing might fail (e.g.,
    URLs with commas, specific datetime cleaning). It produces a Pandas DataFrame
    containing the parsed log data, which is then stored in `session_vars` under
    the explicitly provided `save_as` key. This DataFrame-centric output allows
    the LLM to directly inspect, manipulate, or use the data with other tools.

    Parameters
    ----------
    file_path : str
        Path to the CSV file. Can be absolute or relative to the workspace,
        or a key in `session_vars`.
    expected_datetime_format : str, default "%d/%b/%Y:%H:%M:%S"
        The strptime format string for parsing the timestamp field after
        stripping brackets. E.g., for "[29/Nov/2017:06:58:55]".
    custom_header : List[str] | None, default None
        If the file doesn't have a header or you want to override it, provide
        a list of column names. If None, the first line of the file is used.
    url_column_indices : List[int] | None, default None
        If a column (like a URL) might contain commas, specify its expected
        start and end (exclusive if multiple parts) indices. For a typical weblog.csv like
        IP,Time,URL,Status where URL is parts[2] and parts[3] in a 4-part split due to URL having no commas,
        or parts[2:-1] if URL can have commas. Simplest is to assume URL is between Time and Status.
        This implementation assumes URL is between timestamp_column_index and the status_column_index.
    timestamp_column_index : int, default 1
        The 0-based index of the column containing the timestamp string.
    status_column_is_last : bool, default True
        Indicates if the status column is the last part after splitting by comma.
        If False, url_column_indices needs to be more precise if URL is not just before status.

    save_as : str | None, default None
        The **required** key under which the resulting Pandas DataFrame will be stored
        in `session_vars`. This key must be provided by the caller (LLM).

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the parsed log data.

    Side Effects
    ------------
    - Stores the DataFrame in `session_vars`.
    - Logs detailed errors if parsing fails for specific lines or files.

    Raises
    ------
    FileNotFoundError
        If the specified `file_path` does not exist.
    ValueError
        If `custom_header` is provided but its length doesn't match the number
        of parsed fields, or if other critical parsing issues occur.
    """
    resolved_file_path = _resolve(file_path)
    logger.info(f"Starting custom CSV load for: {resolved_file_path}")

    data = []
    header = custom_header

    try:
        with open(resolved_file_path, 'r') as f:
            if header is None:
                header_line = f.readline().strip()
                if not header_line:
                    msg = f"File {resolved_file_path} is empty or first line (header) is blank."
                    logger.error(msg)
                    raise ValueError(msg)
                header = [h.strip() for h in header_line.split(',')]
                logger.info(f"Using detected header: {header}")

            parsed_header_len = len(header)

            for i, line in enumerate(f, start=1 if header is custom_header else 2):
                line_strip = line.strip()
                if not line_strip:
                    logger.debug(f"Skipping blank line {i} in {resolved_file_path}")
                    continue

                parts = [p.strip() for p in line_strip.split(',')]
                min_expected_parts = timestamp_column_index + 2 # IP, Time, Status at minimum
                if url_column_indices:
                    min_expected_parts = max(min_expected_parts, max(url_column_indices) +1)

                if len(parts) < 2: # Must have at least Time and one other field for basic parsing logic
                    logger.warning(
                        f"Line {i} in {resolved_file_path} has too few parts ({len(parts)}): '{line_strip}'. Skipping."
                    )
                    continue
                
                row_dict = {}
                try:
                    # Basic structure: IP, Time, ..., Status
                    # More robustly: parts[0] is IP, parts[timestamp_column_index] is Time
                    # parts[-1] is Status if status_column_is_last is True
                    # URL is parts[timestamp_column_index+1 : -1 (if status_column_is_last) or specific index]

                    row_dict[header[0]] = parts[0] # Assume first part is always first header col (e.g. IP)
                    
                    time_str_raw = parts[timestamp_column_index]
                    time_str_cleaned = time_str_raw.strip("[]")
                    row_dict[header[timestamp_column_index]] = datetime.strptime(time_str_cleaned, expected_datetime_format)

                    if status_column_is_last:
                        row_dict[header[-1]] = parts[-1]
                        # URL is everything between timestamp and status
                        url_parts = parts[timestamp_column_index+1:-1]
                        if len(header) > timestamp_column_index + 2: # Check if a URL column exists in header
                            row_dict[header[timestamp_column_index+1]] = ','.join(url_parts)
                        elif url_parts: # if no specific URL col name but parts exist
                            logger.warning(f"Line {i}: URL parts '{','.join(url_parts)}' found but no dedicated header column after timestamp and before status. Storing as 'X_URL_inferred'.")
                            row_dict['X_URL_inferred'] = ','.join(url_parts)

                        # Fill other anticipated columns if parts match header length
                        # This part is tricky if URL parts vary. Assuming fixed structure based on header for non-IP,Time,URL,Status
                        if len(parts) == parsed_header_len and parsed_header_len > 3: # More than IP,Time,Status
                            idx_part = timestamp_column_index + 2 # after Time and inferred URL part
                            idx_header = timestamp_column_index + 2 # after Time and inferred URL header
                            while idx_part < len(parts) -1 and idx_header < len(header) -1:
                                row_dict[header[idx_header]] = parts[idx_part]
                                idx_part += 1
                                idx_header +=1
                    else:
                        # TODO: Implement more complex parsing if status is not last or URL indices are explicitly given
                        # This would require careful mapping of parts to header based on url_column_indices
                        logger.warning("Parsing for status_column_is_last=False is not fully implemented yet.")
                        # Fallback: just try to map based on number of parts if it matches header length
                        if len(parts) == parsed_header_len:
                            for h_idx, h_name in enumerate(header):
                                if h_name not in row_dict: # Avoid overwriting already parsed fields
                                    row_dict[h_name] = parts[h_idx]
                        else:
                            logger.warning(f"Line {i}: Parts length {len(parts)} doesn't match header {parsed_header_len} and complex parsing not fully implemented. Partial data for row: '{line_strip}'")                       \

                    data.append(row_dict)

                except ValueError as ve_parse:
                    logger.error(
                        f"Line {i} in {resolved_file_path}: DateTime parsing error for '{time_str_raw}'. "                        f"Expected format: '{expected_datetime_format}'. Error: {ve_parse}. Line: '{line_strip}'"
                    )
                except IndexError as ie_parse:
                    logger.error(
                        f"Line {i} in {resolved_file_path}: Index error, likely malformed CSV structure "                        f"or incorrect column_indices. Parts: {parts}. Error: {ie_parse}. Line: '{line_strip}'"
                    )
                except Exception as e_parse:
                    logger.error(
                        f"Line {i} in {resolved_file_path}: Unexpected error parsing line. "                        f"Error: {e_parse}. Line: '{line_strip}'"
                    )

    except FileNotFoundError:
        logger.error(f"File not found at path: {resolved_file_path}")
        raise
    except Exception as e_file:
        logger.error(f"Failed to read or process file {resolved_file_path}: {e_file}")
        raise

    if not data:
        logger.warning(f"No data was successfully parsed from {resolved_file_path}. Returning empty DataFrame.")
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(data)
        # Reorder columns to match original header if possible, fill missing with NaNs
        if header:
            df = df.reindex(columns=header)
        logger.info(f"Successfully parsed {len(df)} rows into a DataFrame from {resolved_file_path}.")

    _save_as_key = save_as
    if _save_as_key is None:
        base_name = f"{os.path.splitext(os.path.basename(resolved_file_path))[0]}_custom_parsed_df"
        counter = 1
        _save_as_key = f"{base_name}_{counter}"
        while _save_as_key in session_vars:
            counter += 1
            _save_as_key = f"{base_name}_{counter}"
        logger.info(f"\'save_as\' not provided for load_custom_parsed_csv_log. Saving DataFrame to session_vars as '{_save_as_key}'.")
    session_vars[save_as] = df
    logger.info(f"Saved custom parsed DataFrame to session_vars as '{save_as}'.")
    return df


# ---------------------------------------------------------------------------
# Heuristic helper for *dimensions* mapping
# ---------------------------------------------------------------------------


def _default_dimension_mapping(cols: List[str]) -> dict:
    def _norm(name: str) -> str:
        return (
            name.lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

    ts_candidates = [c for c in cols if _norm(c) in {"timestamp", "time", "date", "datetime"}]
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

    return {
        "timestamp": ts_candidates[:1],
        "body": body_candidates[:1] if body_candidates else [],
        "span_id": [],
        "labels": label_candidates[:1],
    }


@app.tool()
@log_tool

def suggest_dimension_mapping(file_path: str, *, save_as: str) -> dict:
    """Generate a heuristic ``dimensions`` mapping for a log file, saved to session_vars.

    Inspects CSV column names and suggests a mapping for "timestamp", "body",
    "labels", etc., useful for `load_file_log_data`. The resulting dictionary
    is stored in `session_vars` under the explicitly provided `save_as` key,
    allowing the LLM to review and use this mapping.

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
        will be stored in `session_vars`. This key must be provided by the caller (LLM).

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
    # get_log_file_columns now requires save_as, but for this internal call,
    # we might not want to pollute session_vars or force the user to name an intermediate.
    # For now, let's assume get_log_file_columns is called by the LLM separately if its output needs saving.
    # This tool's primary purpose is to generate the mapping.
    # Alternative: get_log_file_columns could return cols without saving if save_as is a special marker.
    # For simplicity now, it will save with an auto-generated name if called by suggest_dimension_mapping.
    # This part of get_log_file_columns needs to be revisited if we strictly want NO auto-naming.
    # For now, this tool will save ITS result under the user-provided save_as.
    
    # To avoid get_log_file_columns's save_as, we'd need to duplicate its logic or modify it.
    # Let's assume for now the LLM calls get_log_file_columns first if it wants named columns.
    # This tool will just use pandas directly to get columns for suggestion without saving them itself.
    resolved_file_path = _resolve(file_path)
    try:
        cols = pd.read_csv(resolved_file_path, nrows=0).columns.tolist()
    except Exception as e:
        logger.error(f"Failed to read columns from {resolved_file_path} for suggestion: {e}")
        raise ValueError(f"Could not read columns from {resolved_file_path} to suggest mapping. Error: {e}")

    mapping = _default_dimension_mapping(cols)
    session_vars[save_as] = mapping
    logger.info(f"Saved suggested dimension mapping to session_vars as '{save_as}'.")
    return mapping


# ---------------------------------------------------------------------------
# Combine multiple CSV files
# ---------------------------------------------------------------------------


@app.tool()
@log_tool

def combine_csv_files(
    input_source: Union[str, List[str]],
    file_glob_pattern: str = "*.csv",
    sort_by_columns: Union[List[str], str, None] = None,
    ascending: Union[bool, List[bool]] = True,
    join_axes_how: str = "outer",
    drop_columns: List[str] | None = None,
    *,
    save_as: str,
) -> pd.DataFrame:
    """Read and concatenate multiple CSV files into a single Pandas DataFrame.

    Combines CSV files from a directory (using `file_glob_pattern`) or a list of
    paths/`session_vars` keys. The resulting Pandas DataFrame is stored in
    `session_vars` under the explicitly provided `save_as` key. This allows the
    LLM to work with the combined dataset as a named DataFrame.

    Parameters
    ----------
    input_source : Union[str, List[str]]
        The source of CSV files. This argument is resolved via `_resolve`.
        - If a string: It's treated as a directory path. `file_glob_pattern`
          is used to find files within this directory.
          Example: `"./logs/2023-01/"`
        - If a list: Each item in the list is resolved.
            - If an item resolves to a string, it's treated as a direct file path.
            - If an item resolves to a list of strings, each of those strings is
              treated as a direct file path.
          Example: `["./data/jan.csv", "./data/feb.csv"]` or
                   `["jan_file_var", "feb_file_var"]` (if these are keys in `session_vars`) or
                   `["list_of_files_var"]` (if this key in `session_vars` holds a list of paths).
    file_glob_pattern : str, default "*.csv"
        A glob pattern used to find files when `input_source` is a directory path.
        Ignored if `input_source` is a list.
        Example: `"*.log.csv"`, `"data_*.csv"`.
    sort_by_columns : Union[List[str], str, None], default None
        Column name or list of column names to sort the combined DataFrame by.
        If `None`, no sorting is performed after concatenation, other than the
        implicit order from `glob` if a directory is used.
        Example: `"timestamp"` or `["date", "time"]`.
    ascending : Union[bool, List[bool]], default True
        Sort order for `sort_by_columns`. True for ascending, False for descending.
        If a list, it must match the length of `sort_by_columns`.
    join_axes_how : str, default "outer"
        How to handle columns that are not present in all files during concatenation.
        - "outer": (Default) Keeps all columns from all files. Missing values
          will be `NaN`.
        - "inner": Keeps only columns that are common to all files.
    drop_columns : List[str] | None, default None
        A list of column names to drop from the combined DataFrame before saving
        and returning. Columns are dropped if they exist; no error if a specified
        column is not found.
        Example: `["redundant_col", "temp_id"]`.
    save_as : str
        The **required** key under which the combined Pandas DataFrame will be stored
        in `session_vars`. This key must be provided by the caller (LLM).

    Returns
    -------
    pandas.DataFrame
        A single DataFrame containing data from all combined CSV files.

    Side Effects
    ------------
    - Stores the combined DataFrame in `session_vars` under the key specified by
      `save_as` or an auto-generated key.
    - Raises `ValueError` if `input_source` is invalid, no files are found, or
      `join_axes_how` is not "outer" or "inner".

    Examples
    --------
    # Combine all CSVs in a directory
    >>> combine_csv_files(input_source="./daily_logs/", file_glob_pattern="*.csv", save_as="all_logs")

    # Combine specific files stored in session_vars, then sort
    >>> session_vars["file1"] = "path/to/data1.csv"
    >>> session_vars["file2"] = "path/to/data2.csv"
    >>> combine_csv_files(input_source=["file1", "file2"], sort_by_columns="timestamp", save_as="sorted_data")

    Notes
    -----
    - The `input_source` argument (and its list elements, if applicable) are
      resolved using `_resolve`.
    - File paths are sorted alphabetically when read using `glob` from a directory
      before concatenation, which might affect order if not explicitly sorted later.
    """
    resolved_input_source = _resolve(input_source)

    actual_file_paths: List[str] = []

    if isinstance(resolved_input_source, str):
        directory_path = resolved_input_source
        if not os.path.isdir(directory_path):
            raise ValueError(
                f"Input source '{directory_path}' is a string but not a valid directory."
            )
        glob_path = os.path.join(directory_path, file_glob_pattern)
        actual_file_paths.extend(sorted(glob.glob(glob_path)))
    elif isinstance(resolved_input_source, list):
        for item in resolved_input_source:
            resolved = _resolve(item)
            if isinstance(resolved, list):
                actual_file_paths.extend(map(str, resolved))
            elif isinstance(resolved, str):
                actual_file_paths.append(resolved)
            else:
                raise ValueError(
                    "combine_csv_files: items must resolve to str or List[str]"
                )
    else:
        raise ValueError(
            "'input_source' must be a directory path or list of paths / session vars."
        )

    if not actual_file_paths:
        raise ValueError("No CSV files found to combine.")

    if join_axes_how not in {"outer", "inner"}:
        raise ValueError("join_axes_how must be 'outer' or 'inner'.")

    dfs = []
    for path in actual_file_paths:
        dfs.append(pd.read_csv(path))

    combined_df = pd.concat(dfs, join=join_axes_how, ignore_index=True)  # type: ignore[arg-type]

    if drop_columns:
        combined_df = combined_df.drop(columns=[c for c in drop_columns if c in combined_df.columns])

    if sort_by_columns:
        combined_df = combined_df.sort_values(by=sort_by_columns, ascending=ascending, ignore_index=True)

    _save_as_key = save_as
    if _save_as_key is None:
        base_name = "combined_csv_dataframe"
        counter = 1
        _save_as_key = f"{base_name}_{counter}"
        while _save_as_key in session_vars:
            counter += 1
            _save_as_key = f"{base_name}_{counter}"
        logger.info(f"'save_as' not provided for combine_csv_files. Saving result to session_vars as '{_save_as_key}'.")
    session_vars[save_as] = combined_df
    logger.info(f"Saved combined CSV DataFrame to session_vars as '{save_as}'.")
    return combined_df


# ---------------------------------------------------------------------------
# LogAI OpenSet loader
# ---------------------------------------------------------------------------


@app.tool()
@log_tool

def load_openset_log_data(
    dataset_name: str,
    data_dir: str = "../datasets",
    *,
    save_as: str,
) -> Any: # Returns LogRecordObject
    """Load a LogAI open dataset (e.g., HDFS, BGL) as a LogRecordObject.

    Loads a standard LogAI open dataset into a `LogRecordObject`, which is then
    stored in `session_vars` under the explicitly provided `save_as` key.
    This `LogRecordObject` can then be used by other LogAI tools. The LLM is
    expected to name this output.

    Parameters
    ----------
    dataset_name : str
        The name of the open dataset to load. This name is used to determine the
        subdirectory and filename within `data_dir`.
        Common examples: "HDFS", "BGL", "Thunderbird", "Spirit", "Liberty", "HealthApp".
        This argument is **not** resolved from `session_vars`.
    data_dir : str, default "../datasets"
        The base directory where the open datasets are stored. Each dataset is
        expected to be in a subdirectory named after `dataset_name` (e.g.,
        `../datasets/HDFS/`). The log file itself is typically named
        `{dataset_name}.log` (e.g., `HDFS.log`), with a special case for
        "HealthApp" which uses `HealthApp_2000.log`.
        This argument is **not** resolved from `session_vars`.
    save_as : str
        The **required** key under which the loaded `LogRecordObject` will be stored
        in `session_vars`. This key must be provided by the caller (LLM).

    Returns
    -------
    logai.dataloader.data_structures.LogRecordObject
        An object containing the loaded and structured log data from the open dataset.

    Side Effects
    ------------
    - Stores the `LogRecordObject` in `session_vars` under the key specified by
      `save_as` or an auto-generated key.
    - Raises `FileNotFoundError` if the dataset file cannot be found at the
      expected path (constructed from `data_dir`, `dataset_name`, and the
      conventional filename).

    Examples
    --------
    >>> load_openset_log_data(dataset_name="HDFS", save_as="hdfs_data")
    # session_vars["hdfs_data"] will contain the LogRecordObject for HDFS dataset.

    # Assuming BGL dataset is in '../datasets/BGL/BGL.log'
    >>> load_openset_log_data(dataset_name="BGL")
    # session_vars["BGL_log_record_1"] (or similar) will store the data.

    See Also
    --------
    load_file_log_data : For loading custom log files not part of open datasets.

    Notes
    -----
    - The availability of datasets depends on the LogAI setup and whether the
      specified `data_dir` correctly points to them.
    - This tool does not resolve `dataset_name` or `data_dir` from `session_vars`.
    """
    if dataset_name == "HealthApp":
        filename = f"{dataset_name}_2000.log"
    else:
        filename = f"{dataset_name}.log"

    full_path = os.path.abspath(os.path.join(data_dir, dataset_name, filename))
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found at expected path: {full_path}"
        )

    cfg = OpenSetDataLoaderConfig()
    cfg.dataset_name = dataset_name
    cfg.filepath = full_path
    loader = OpenSetDataLoader(cfg)
    log_record = loader.load_data()

    _save_as_key = save_as
    if _save_as_key is None:
        base_name = f"{dataset_name}_log_record"
        counter = 1
        _save_as_key = f"{base_name}_{counter}"
        while _save_as_key in session_vars:
            counter += 1
            _save_as_key = f"{base_name}_{counter}"
        logger.info(f"'save_as' not provided for load_openset_log_data. Saving result to session_vars as '{_save_as_key}'.")
    session_vars[save_as] = log_record
    logger.info(f"Saved loaded open dataset LogRecordObject to session_vars as '{save_as}'.")
    return log_record
