# dataframe_utils.py
# Utilities for converting MCP responses and handling DataFrames (Polars/Pandas)

import polars as pl
import pandas as pd
from pandas import json_normalize
from logai.dataloader.data_model import LogRecordObject
from typing import Any, Dict, List, Union, Optional

def as_polars_df(mcp_response: Union[Dict[str, Any], List[Dict[str, Any]]], result_key: str = "result") -> pl.DataFrame:
    """
    Convert an MCP JSON-RPC response (or a direct list/dict) into a Polars DataFrame.

    Args:
        mcp_response: The MCP response, typically a dictionary with a 'result' key
                      containing a list of dictionaries, or a direct list of dictionaries,
                      or a single dictionary.
        result_key: The key in the MCP response that holds the actual data.

    Returns:
        A Polars DataFrame.

    Raises:
        ValueError: If the response data cannot be converted to a Polars DataFrame.
    """
    data_to_convert: Union[List[Dict[str, Any]], Dict[str, Any]]

    if isinstance(mcp_response, dict) and result_key in mcp_response:
        data_to_convert = mcp_response[result_key]
    else:
        data_to_convert = mcp_response

    if isinstance(data_to_convert, list) and all(isinstance(row, dict) for row in data_to_convert):
        if not data_to_convert: # Handle empty list
            return pl.DataFrame()
        return pl.from_dicts(data_to_convert)
    elif isinstance(data_to_convert, dict): # Handle single record result
        return pl.from_dicts([data_to_convert])
    elif isinstance(data_to_convert, list) and not data_to_convert: # Handle empty list directly passed
        return pl.DataFrame()


    raise ValueError(
        f"Input data is not a list of dicts, a single dict, or an empty list; "
        f"cannot convert to Polars DataFrame. Type received: {type(data_to_convert)}"
    )


def mcp_to_pandas_df(
    mcp_response: Union[Dict[str, Any], List[Dict[str, Any]]],
    result_key: str = "result",
    normalize_nested: bool = True,
    meta: Optional[List[Union[str, List[str]]]] = None, # Updated type hint
    record_path: Optional[Union[str, List[str]]] = None,
    # Additional json_normalize params can be added here if needed
) -> pd.DataFrame:
    """
    Convert an MCP JSON-RPC response (or a direct list/dict) into a Pandas DataFrame.
    Handles simple list of dicts and can normalize nested structures.

    Args:
        mcp_response: The MCP response, typically a dictionary with a 'result' key
                      containing a list of dictionaries, or a direct list of dictionaries,
                      or a single dictionary.
        result_key: The key in the MCP response that holds the actual data.
        normalize_nested: If True (default), uses pandas.json_normalize to flatten
                          nested dictionaries. If False, uses a simple pd.DataFrame
                          conversion which might result in columns containing dicts/lists.
        meta: Argument for json_normalize: List of fields to include from the top-level
              dictionary in each record of the normalized DataFrame.
        record_path: Argument for json_normalize: Path in each object to list of records.
                     Use if data is nested under another key within each item of the list.

    Returns:
        A Pandas DataFrame.

    Raises:
        ValueError: If the response data cannot be converted to a Pandas DataFrame.
    """
    data_to_convert: Union[List[Dict[str, Any]], Dict[str, Any]]

    if isinstance(mcp_response, dict) and result_key in mcp_response:
        data_to_convert = mcp_response[result_key]
    elif isinstance(mcp_response, (list, dict)):
        data_to_convert = mcp_response
    else:
        raise ValueError(
            f"Input mcp_response is not a dict or list. Type received: {type(mcp_response)}"
        )

    if not data_to_convert and isinstance(data_to_convert, list): # Handle empty list
        return pd.DataFrame()
    
    if isinstance(data_to_convert, dict): # Handle single record result, wrap in list
        data_to_convert = [data_to_convert]

    if isinstance(data_to_convert, list) and all(isinstance(row, dict) for row in data_to_convert):
        if normalize_nested:
            try:
                return json_normalize(data_to_convert, record_path=record_path, meta=meta)
            except Exception as e:
                # Fallback or re-raise with more context
                # Sometimes json_normalize fails if data is too irregular or not list of dicts
                # For now, try simple conversion if normalize fails with common issues
                try:
                    return pd.DataFrame(data_to_convert)
                except Exception as e_simple:
                    raise ValueError(
                        f"Failed to convert to Pandas DataFrame with json_normalize (error: {e}) "
                        f"and simple pd.DataFrame (error: {e_simple}). "
                        f"Data type: {type(data_to_convert)}"
                    ) from e_simple
        else:
            return pd.DataFrame(data_to_convert)

    raise ValueError(
        f"Data to convert is not a list of dicts or a single dict. "
        f"Type received: {type(data_to_convert)}"
    )

def convert_to_logai_input(
    polars_df: pl.DataFrame,
    log_record_meta: Dict[str, List[str]],
    default_log_body_column_name: str = "logline"
) -> LogRecordObject:
    """
    Converts a Polars DataFrame to a logai.LogRecordObject.

    This involves an intermediate conversion to a Pandas DataFrame, as LogRecordObject
    is designed to be created from Pandas DataFrames.

    Args:
        polars_df: The Polars DataFrame to convert.
        log_record_meta: A dictionary mapping LogRecordObject field names
                         (e.g., 'body', 'timestamp', 'attributes') to lists of
                         column names in the Polars DataFrame that should populate
                         those fields.
        default_log_body_column_name: The name to use for the 'body' column in the
                                      Pandas DataFrame if 'body' in log_record_meta
                                      points to multiple columns that need to be joined,
                                      or a single column that needs renaming.

    Returns:
        A LogRecordObject instance.

    Raises:
        KeyError: If a column specified in log_record_meta is not found in the Polars DataFrame.
        ValueError: If log_record_meta is empty or doesn't conform to expectations.
    """
    if not log_record_meta:
        raise ValueError("log_record_meta cannot be empty. Please specify column mappings.")

    # Convert Polars DataFrame to Pandas DataFrame
    try:
        pandas_df = polars_df.to_pandas()
    except Exception as e:
        raise RuntimeError(f"Failed to convert Polars DataFrame to Pandas DataFrame: {e}")

    # Prepare a new Pandas DataFrame based on the meta mapping for LogRecordObject
    # LogRecordObject.from_dataframe expects specific column names in the input pandas_df
    # that match the 'value' part of its own meta_data argument.
    # So, we need to construct a pandas_df that has columns named according to logai's expectations.

    # Example: if log_record_meta = {"body": ["message_col"], "timestamp": ["time_col"]}
    # and polars_df has columns "message_col", "time_col"
    # The pandas_df passed to LogRecordObject.from_dataframe should have columns that
    # LogRecordObject.from_dataframe can pick using its own meta_data.
    # The meta_data for LogRecordObject.from_dataframe looks like:
    # {'body': ['logline'], 'timestamp': ['timestamp_col_in_df']}

    # Let's simplify: LogRecordObject.from_dataframe takes the full pandas_df
    # and a meta_data dict that tells it WHICH columns from THAT pandas_df map to ITS fields.
    # So, the pandas_df we pass should contain all necessary columns, and the
    # meta_data for from_dataframe should correctly point to them.

    # Ensure all specified columns in log_record_meta exist in the pandas_df
    for field_name, col_names in log_record_meta.items():
        for col_name in col_names:
            if col_name not in pandas_df.columns:
                raise KeyError(
                    f"Column '{col_name}' specified in log_record_meta for field '{field_name}' "
                    f"not found in the DataFrame."
                )

    # The meta_data for LogRecordObject.from_dataframe maps LogRecordObject fields
    # to the column names *as they exist in the pandas_df we are passing to it*.
    # So, log_record_meta itself can serve as this mapping if column names are already suitable.
    # However, LogRecordObject has specific expectations for 'body' (often needs to be a single column).

    # Create the LogRecordObject
    try:
        # The `meta_data` argument for `from_dataframe` maps LogRecordObject fields
        # to the list of column names in the `pandas_df` that should populate that field.
        log_object = LogRecordObject.from_dataframe(data=pandas_df, meta_data=log_record_meta)
    except Exception as e:
        # Provide more context on failure
        raise RuntimeError(
            f"Failed to create LogRecordObject from Pandas DataFrame. Error: {e}. "
            f"Ensure log_record_meta correctly maps to columns in the DataFrame: {pandas_df.head()} "
            f"with meta: {log_record_meta}"
        )

    return log_object
