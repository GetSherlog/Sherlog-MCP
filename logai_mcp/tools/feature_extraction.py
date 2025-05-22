"""Feature extraction utilities (semantic + timeseries).

Refactored to follow the shell-execution pattern:
- Core logic in `_extract_log_features_impl` and `_extract_timeseries_features_impl`.
- Public wrappers assign results to `save_as` names in the shell.
- No direct `session_vars` or `_resolve` in helpers.
"""

from typing import Any, Dict, List, Union

import pandas as pd
import numpy as np

from logai_mcp.session import (
    app,
    logger,
)

from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor

from logai_mcp.ipython_shell_utils import _SHELL, run_code_in_shell

def _extract_log_features_impl(
    log_vectors: Union[pd.Series, pd.DataFrame, np.ndarray, list[float]],
    attributes_encoded: Union[pd.DataFrame, np.ndarray, list[float]],
    timestamps: Union[pd.Series, np.ndarray, list[float]],
    max_feature_len: int = 100,
) -> pd.DataFrame:
    """Combine log vectors, attributes, and timestamps into a unified feature DataFrame.

    Integrates numerical log vectors, encoded categorical attributes, and timestamps
    into a single feature vector/DataFrame (Pandas/NumPy). This function contains
    the core logic, while the public wrapper `extract_log_features` handles
    interaction with the shell environment, such as saving the output.

    Parameters
    ----------
    log_vectors : Union[pd.Series, pd.DataFrame, np.ndarray, list[float]]
        Input data for log vectors. Can be a pandas Series, DataFrame,
        NumPy array, or list of floats. Typically, these are numerical
        representations of log content (e.g., from Word2Vec, TF-IDF).
        Providing a pandas Series is often suitable.
    attributes_encoded : Union[pd.DataFrame, np.ndarray, list[float]]
        Input data for encoded attributes. Can be a pandas DataFrame,
        NumPy array, or list of floats (e.g., list of lists for
        multi-attribute encoding). Typically, these are numerically
        encoded categorical attributes of the logs. Providing a pandas
        DataFrame is often suitable.
    timestamps : Union[pd.Series, np.ndarray, list[float]]
        Input data for timestamps. Can be a pandas Series, NumPy array,
        or list of floats/timestamp-like objects. These correspond to
        the log entries. Providing a pandas Series is often suitable.
    max_feature_len : int, default 100
        The maximum length for the feature vector, used by LogAI's `FeatureExtractor`.
        This parameter might influence truncation or padding if applicable within LogAI.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the combined feature vectors.

    Side Effects
    ------------
    This function does not directly interact with or modify shell variables.
    The calling wrapper function (`extract_log_features`) is responsible for
    any such side effects (e.g., saving results to the shell).

    See Also
    --------
    vectorize_log_data : Provides `log_vectors`.
    encode_log_attributes : Provides `attributes_encoded`.
    load_file_log_data : Can provide `timestamps` (e.g., from `log_record.timestamp`).
    detect_semantic_anomalies : Might consume the output of this tool.
    cluster_log_features : Might consume the output of this tool.
    extract_log_features : The public wrapper for this implementation.

    Notes
    -----
    - Inputs like `log_vectors`, `attributes_encoded`, and `timestamps` are
      processed and, if necessary, converted to pandas Series/DataFrames as
      required by LogAI's `FeatureExtractor`. It is recommended to ensure
      inputs are already in appropriate pandas types with compatible indices
      to avoid unexpected behavior or data misalignment.
    - This function utilizes `logai.information_extraction.feature_extractor.FeatureExtractor`
      for the core feature extraction logic.
    """

    if not isinstance(log_vectors, (pd.Series, np.ndarray, list)):
        raise TypeError(f"log_vectors must be Series, ndarray or list, got {type(log_vectors)}")
    log_series = pd.Series(log_vectors) if not isinstance(log_vectors, pd.Series) else log_vectors

    if isinstance(attributes_encoded, pd.DataFrame):
        attrs_df = attributes_encoded
    else:
        attrs_df = pd.DataFrame(attributes_encoded)
    ts_series = pd.Series(timestamps) if not isinstance(timestamps, pd.Series) else timestamps

    if not (log_series.index.equals(ts_series.index) and 
            (attrs_df.empty or log_series.index.equals(attrs_df.index))):
        logger.warning(
            "Indices of log_vectors, timestamps, and attributes_encoded (if not empty) do not match. "
            "This may lead to misaligned features or errors in FeatureExtractor. "
            f"Log/TS length: {len(log_series)}/{len(ts_series)}. Attrs length: {len(attrs_df)}. "
            f"Log index type: {type(log_series.index)}, TS index type: {type(ts_series.index)}, Attrs index type: {type(attrs_df.index) if not attrs_df.empty else 'N/A'}"
        )

    cfg = FeatureExtractorConfig()
    cfg.max_feature_len = max_feature_len
    extractor = FeatureExtractor(cfg)
    
    try:
        feat_vec = extractor.convert_to_feature_vector(log_series, attrs_df, ts_series)
    except Exception as e:
        logger.error(
            f"Error during LogAI FeatureExtractor.convert_to_feature_vector: {e}.\n"
            f"  Log series shape: {log_series.shape}, type: {type(log_series)}\n"
            f"  Attributes DF shape: {attrs_df.shape}, type: {type(attrs_df)}\n"
            f"  Timestamp series shape: {ts_series.shape}, type: {type(ts_series)}"
        )
        raise RuntimeError(f"LogAI FeatureExtractor failed. Original error: {e}") from e

    return feat_vec

_SHELL.push({"_extract_log_features_impl": _extract_log_features_impl})

@app.tool()
async def extract_log_features(
    log_vectors: Union[str, pd.Series, pd.DataFrame, np.ndarray, list[float]],
    attributes_encoded: Union[str, pd.DataFrame, np.ndarray, list[float]],
    timestamps: Union[str, pd.Series, np.ndarray, list[float]],
    max_feature_len: int = 100,
    *,
    save_as: str,
):
    """Wrapper for `_extract_log_features_impl`.

    Assigns the combined feature vector/DataFrame to `save_as` in the shell.
    Input arguments `log_vectors`, `attributes_encoded`, `timestamps` can be
    variable names (strings) or actual data objects.
    """
    lv_arg = log_vectors if isinstance(log_vectors, str) else repr(log_vectors)
    ae_arg = attributes_encoded if isinstance(attributes_encoded, str) else repr(attributes_encoded)
    ts_arg = timestamps if isinstance(timestamps, str) else repr(timestamps)

    code = (
        f"{save_as} = _extract_log_features_impl({lv_arg}, {ae_arg}, {ts_arg}, {max_feature_len})\n"
    )
    return await run_code_in_shell(code)

extract_log_features.__doc__ = _extract_log_features_impl.__doc__

def _extract_timeseries_features_impl(
    parsed_loglines: pd.Series,
    attributes: pd.DataFrame,
    timestamps: pd.Series,
    group_by_time: str,
    group_by_category: List[str],
    feature_extractor_params: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Convert logs into counter vectors (Pandas DataFrame) for time-series analysis.

    Processes parsed log lines (Pandas Series), attributes (Pandas DataFrame),
    and timestamps (Pandas Series) to generate time-bucketed event counts,
    grouped by specified categories. Returns a Pandas DataFrame of counter
    vectors suitable for time-series anomaly detection.

    This function is the core implementation logic. The `extract_timeseries_features`
    tool wraps this function to handle argument resolution from the shell and
    saving the result.

    Parameters
    ----------
    parsed_loglines : pd.Series
        Parsed log templates or messages.
    attributes : pd.DataFrame
        Attributes associated with each log line (e.g., hostname, application ID).
    timestamps : pd.Series
        Timestamps for each log line.
    group_by_time : str
        The time window to group events by (e.g., "5s", "1min", "1H"). This uses
        pandas time frequency strings.
    group_by_category : List[str]
        A list of column names from the `attributes` DataFrame to use for further
        grouping the time-bucketed events. For each unique combination of these
        categories, a separate time series of counts will be generated.
        Example: `["hostname", "error_code"]`.
    feature_extractor_params : Dict[str, Any] | None, default None
        Additional parameters for LogAI's `FeatureExtractorConfig`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where rows typically represent time windows and groups, and
        columns include counts and group identifiers. The exact structure is
        determined by LogAI's `FeatureExtractor.convert_to_counter_vector`.

    Examples
    --------
    The following example demonstrates usage via the wrapper tool, which is how
    this implementation is typically invoked:
    
    # Assuming "templates", "attrs", "ts" are appropriate data in session_vars:
    >>> extract_timeseries_features(
    ...     parsed_loglines="templates",
    ...     attributes="attrs",
    ...     timestamps="ts",
    ...     group_by_time="10min",
    ...     group_by_category=["host"],
    ...     save_as="host_event_counts_10min"
    ... )
    # session_vars["host_event_counts_10min"] will store the counter DataFrame.

    See Also
    --------
    extract_timeseries_features : The wrapper tool that calls this function.
    parse_log_data : Provides `parsed_loglines`.
    preprocess_log_data : Can provide `attributes` (from its result tuple).
    load_file_log_data : Can provide `timestamps` (from `log_record.timestamp`).
    detect_timeseries_anomalies : Typically consumes the output of this tool.

    Notes
    -----
    - This function directly expects pandas Series/DataFrame objects as inputs.
    - Uses `logai.information_extraction.feature_extractor.FeatureExtractor` with
      its `convert_to_counter_vector` method.
    """

    cfg = FeatureExtractorConfig()
    cfg.group_by_time = group_by_time
    cfg.group_by_category = group_by_category
    if feature_extractor_params:
        for k, v in feature_extractor_params.items():
            setattr(cfg, k, v)
            
    extractor = FeatureExtractor(cfg)
    counter_df = extractor.convert_to_counter_vector(parsed_loglines, attributes, timestamps)

    return counter_df

_SHELL.push({"_extract_timeseries_features_impl": _extract_timeseries_features_impl})

@app.tool()
async def extract_timeseries_features(
    parsed_loglines: Union[str, pd.Series],
    attributes: Union[str, pd.DataFrame],
    timestamps: Union[str, pd.Series],
    group_by_time: str,
    group_by_category: List[str],
    feature_extractor_params: Dict[str, Any] | None = None,
    *,
    save_as: str,
):
    """Wrapper for `_extract_timeseries_features_impl`.

    Assigns the counter vector DataFrame to `save_as` in the shell.
    Inputs `parsed_loglines`, `attributes`, `timestamps` can be variable names or objects.
    """
    pl_arg = parsed_loglines if isinstance(parsed_loglines, str) else repr(parsed_loglines)
    attrs_arg = attributes if isinstance(attributes, str) else repr(attributes)
    ts_arg = timestamps if isinstance(timestamps, str) else repr(timestamps)

    code = (
        f"{save_as} = _extract_timeseries_features_impl({pl_arg}, {attrs_arg}, {ts_arg}, "
        f"{repr(group_by_time)}, {repr(group_by_category)}, {repr(feature_extractor_params)})\n"
    )
    return await run_code_in_shell(code)

extract_timeseries_features.__doc__ = _extract_timeseries_features_impl.__doc__
