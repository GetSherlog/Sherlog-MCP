"""DataFrame compatibility utilities for LogAI MCP.

This module provides a bridge between polars and pandas DataFrames,
allowing the codebase to use polars for performance while maintaining
pandas compatibility for LogAI library integration.
"""

import logging
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

DataFrame = pl.DataFrame
LazyFrame = pl.LazyFrame


def to_polars(df: pd.DataFrame | pl.DataFrame | Any) -> pl.DataFrame:
    """Convert a DataFrame to polars format.

    Args:
        df: Input DataFrame (pandas or polars)

    Returns:
        pl.DataFrame: Polars DataFrame

    Raises:
        ImportError: If polars is not available
        ValueError: If conversion fails

    """
    if isinstance(df, pl.DataFrame):
        return df
    elif isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    else:
        raise ValueError(f"Cannot convert {type(df)} to polars DataFrame")


def to_pandas(df: pd.DataFrame | pl.DataFrame | Any) -> pd.DataFrame:
    """Convert a DataFrame to pandas format.

    Args:
        df: Input DataFrame (pandas or polars)

    Returns:
        pd.DataFrame: Pandas DataFrame

    """
    if isinstance(df, pd.DataFrame):
        return df
    elif isinstance(df, pl.DataFrame):
        return df.to_pandas()
    else:
        raise ValueError(f"Cannot convert {type(df)} to pandas DataFrame")


def create_dataframe_polars(data: Any) -> pl.DataFrame:
    """Create a polars DataFrame from various data formats.

    Args:
        data: Input data in various formats

    Returns:
        pl.DataFrame: Created DataFrame

    """
    if data is None:
        return pl.DataFrame()

    # Already a polars DataFrame
    if isinstance(data, pl.DataFrame):
        return data

    # Convert from pandas
    if isinstance(data, pd.DataFrame):
        return pl.from_pandas(data)

    # List of dictionaries (most common case)
    if isinstance(data, list) and len(data) > 0:
        if all(isinstance(item, dict) for item in data):
            return pl.DataFrame(data)
        # List of lists with potential header
        elif all(isinstance(item, list) for item in data):
            if len(data) > 1:
                try:
                    # Try with first row as header
                    return pl.DataFrame(data[1:], schema=data[0])
                except:
                    # Fall back to no header
                    return pl.DataFrame(data)
            else:
                return pl.DataFrame(data)

    # Dictionary with list values (column-oriented data)
    if isinstance(data, dict):
        # Check if all values are lists of same length
        if all(isinstance(v, list) for v in data.values()):
            lengths = [len(v) for v in data.values()]
            if len(set(lengths)) == 1:  # All same length
                return pl.DataFrame(data)

        # Single row dictionary
        if all(not isinstance(v, (list, dict)) for v in data.values()):
            return pl.DataFrame([data])

    # NumPy array
    if isinstance(data, np.ndarray):
        return pl.DataFrame(data)

    # String that might be CSV or JSON
    if isinstance(data, str):
        # Try JSON first
        try:
            import json

            json_data = json.loads(data)
            return create_dataframe_polars(json_data)  # Recursive call
        except:
            pass

        # Try CSV
        try:
            return pl.read_csv(StringIO(data))
        except:
            pass

    # If all else fails, create empty DataFrame
    logger.warning(
        f"Could not convert {type(data)} to polars DataFrame, returning empty DataFrame"
    )
    return pl.DataFrame()


def create_dataframe_pandas(data: Any) -> pl.DataFrame | pd.DataFrame | Any:
    """Create a pandas DataFrame from various data formats.

    This is a fallback for when polars is not available or when
    pandas is specifically required.

    Args:
        data: Input data in various formats

    Returns:
        pd.DataFrame: Created DataFrame

    """
    if data is None:
        return pd.DataFrame()

    # Already a pandas DataFrame
    if isinstance(data, pd.DataFrame):
        return data

    # Convert from polars
    if isinstance(data, pl.DataFrame):
        return data.to_pandas()

    # List of dictionaries (most common case)
    if isinstance(data, list) and len(data) > 0:
        if all(isinstance(item, dict) for item in data):
            return pd.DataFrame(data)
        # List of lists with potential header
        elif all(isinstance(item, list) for item in data):
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
            return create_dataframe_pandas(json_data)
        except:
            pass

        # Try CSV
        try:
            return pd.read_csv(StringIO(data))
        except:
            pass
    return data


def smart_create_dataframe(
    data: Any, prefer_polars: bool = True
) -> pl.DataFrame | pd.DataFrame:
    """Intelligently create a DataFrame, preferring polars when available.

    Args:
        data: Input data in various formats
        prefer_polars: Whether to prefer polars over pandas

    Returns:
        DataFrame: Created DataFrame (polars or pandas)

    """
    if prefer_polars:
        try:
            return create_dataframe_polars(data)
        except Exception as e:
            logger.warning(
                f"Failed to create polars DataFrame: {e}, falling back to pandas"
            )
            return create_dataframe_pandas(data)
    else:
        return create_dataframe_pandas(data)


def read_csv_smart(file_path: str, **kwargs) -> DataFrame:
    """Smart CSV reader that uses polars when available for better performance.

    Args:
        file_path: Path to CSV file
        **kwargs: Additional arguments passed to the reader

    Returns:
        DataFrame: Loaded DataFrame

    """
    return pl.read_csv(file_path, **kwargs)


def ensure_pandas_for_logai(df: DataFrame) -> pd.DataFrame:
    """Ensure DataFrame is in pandas format for LogAI library compatibility.

    This function should be used before passing DataFrames to LogAI components.

    Args:
        df: Input DataFrame

    Returns:
        pd.DataFrame: Pandas DataFrame ready for LogAI

    """
    return to_pandas(df)


def optimize_for_analytics(df: DataFrame) -> DataFrame:
    """Optimize DataFrame for analytics operations.

    Converts to polars if available for better performance in analytical operations.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame: Optimized DataFrame

    """
    return to_polars(df)


def to_json_serializable(
    df: pl.DataFrame | pd.DataFrame | Any,
) -> list[dict[str, Any]] | Any:
    """Convert DataFrame to JSON-serializable format for MCP tool responses.

    This function properly handles pandas and polars DataFrames to ensure
    they are correctly serialized for JSON responses, avoiding issues with
    special data types, timestamps, and NaN values.

    Args:
        df: Input DataFrame or other data

    Returns:
        JSON-serializable data (list of dicts for DataFrames, original data otherwise)

    """
    import json

    if isinstance(df, pd.DataFrame):
        return json.loads(df.to_json(orient="records", date_format="iso"))
    elif isinstance(df, pl.DataFrame):
        return df.to_dicts()
    else:
        return df


# Export commonly used functions
__all__ = [
    "DataFrame",
    "LazyFrame",
    "to_polars",
    "to_pandas",
    "smart_create_dataframe",
    "read_csv_smart",
    "ensure_pandas_for_logai",
    "optimize_for_analytics",
    "to_json_serializable",
]
