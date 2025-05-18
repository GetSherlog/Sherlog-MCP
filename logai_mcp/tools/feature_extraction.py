"""Feature extraction utilities (semantic + timeseries)."""

from typing import Any, Dict, List

import pandas as pd
import numpy as np

from logai_mcp.session import (
    app,
    log_tool,
    _resolve,
    session_vars,
    logger,
)

from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor
from logai.utils import constants

# ---------------------------------------------------------------------------
# Concatenate semantic features
# ---------------------------------------------------------------------------


@app.tool()
@log_tool
def extract_log_features(
    log_vectors: Any,
    attributes_encoded: Any,
    timestamps: Any,
    max_feature_len: int = 100,
    *,
    save_as: str,
):
    """Combine log vectors, attributes, and timestamps into a unified feature set.

    Integrates numerical log vectors, encoded categorical attributes, and timestamps
    into a single feature vector/DataFrame (Pandas/NumPy). This output is
    **mandatorily** stored in `session_vars` under the key provided in `save_as`,
    making it ready for ML model input. The LLM is expected to name this feature set.

    Parameters
    ----------
    log_vectors : Any
        Numerical vectors representing log content (e.g., from Word2Vec, TF-IDF).
        This argument is resolved from `session_vars` if it's a string key.
        Expected to be a pandas Series or a type readily convertible to it (like list or NumPy array).
        Providing a pandas Series is recommended.
        Example: `"log_vectors_w2v_var"` (a key in `session_vars`).
    attributes_encoded : Any
        Numerically encoded categorical attributes of the logs.
        This argument is resolved from `session_vars` if it's a string key.
        Expected to be a pandas DataFrame or a type readily convertible to it (like NumPy 2D array).
        Providing a pandas DataFrame is recommended.
        Example: `"encoded_attributes_var"` (a key in `session_vars`).
    timestamps : Any
        Timestamps corresponding to the log entries.
        This argument is resolved from `session_vars` if it's a string key.
        Expected to be a pandas Series or a type readily convertible to it (like list).
        Providing a pandas Series is recommended.
        Example: `"log_timestamps_series_var"` (a key in `session_vars`).
    max_feature_len : int, default 100
        The maximum length for the feature vector, used by LogAI's `FeatureExtractor`.
        This parameter might influence truncation or padding if applicable within LogAI.
        This argument is **not** resolved from `session_vars`.
    save_as : str
        The **required** key under which the extracted feature vector/DataFrame
        will be stored in `session_vars`. Must be provided by the caller (LLM).

    Returns
    -------
    Any (typically pandas.DataFrame or numpy.ndarray)
        The combined feature vector. The exact type depends on the LogAI
        `FeatureExtractor` implementation but is often a NumPy array or pandas DataFrame.

    Side Effects
    ------------
    - Stores the resulting feature vector/DataFrame in `session_vars` under the key
      specified by `save_as` or an auto-generated key.

    Examples
    --------
    # Assuming "semantic_vectors", "encoded_attrs", "ts_data" are in session_vars:
    >>> extract_log_features(
    ...     log_vectors="semantic_vectors",
    ...     attributes_encoded="encoded_attrs",
    ...     timestamps="ts_data",
    ...     save_as="combined_features"
    ... )
    # session_vars["combined_features"] will hold the unified feature set.

    See Also
    --------
    vectorize_log_data : Provides `log_vectors`.
    encode_log_attributes : Provides `attributes_encoded`.
    load_file_log_data : Can provide `timestamps` (e.g., from `log_record.timestamp`).
    detect_semantic_anomalies : Might consume the output of this tool.
    cluster_log_features : Might consume the output of this tool.

    Notes
    -----
    - All primary data inputs (`log_vectors`, `attributes_encoded`, `timestamps`)
      are resolved using `_resolve`.
    - Inputs are converted to pandas Series/DataFrames as expected by LogAI's
      `FeatureExtractor`. It is recommended to ensure inputs are already in the
      correct pandas types with compatible indices to avoid unexpected behavior.
    - Uses `logai.information_extraction.feature_extractor.FeatureExtractor`.
    """

    lv_resolved = _resolve(log_vectors)
    ae_resolved = _resolve(attributes_encoded)
    ts_resolved = _resolve(timestamps)

    # Input validation and conversion
    try:
        if lv_resolved is None:
            raise ValueError("Resolved 'log_vectors' is None. A pandas Series or convertible type is required.")
        log_series = pd.Series(lv_resolved) if not isinstance(lv_resolved, pd.Series) else lv_resolved
        if log_series.empty:
            logger.warning("'log_vectors' resolved to an empty Series.")
    except Exception as e:
        logger.error(f"Error converting 'log_vectors' to pandas Series: {e}. Input was: {lv_resolved}")
        raise TypeError(f"'log_vectors' could not be converted to a pandas Series. Error: {e}") from e

    try:
        if ae_resolved is None:
            # LogAI's FeatureExtractor can sometimes handle None attributes, but being explicit is good.
            logger.info("Resolved 'attributes_encoded' is None. Proceeding with None attributes.")
            attrs_df = pd.DataFrame() # Pass an empty DataFrame if attributes are None
        elif isinstance(ae_resolved, pd.DataFrame):
            attrs_df = ae_resolved
        else:
            attrs_df = pd.DataFrame(ae_resolved)
        
        if ae_resolved is not None and attrs_df.empty and not isinstance(ae_resolved, pd.DataFrame) and len(ae_resolved) > 0:
             logger.warning("'attributes_encoded' (not None) resolved to an empty DataFrame, but original input was not empty. Check conversion.")
        elif ae_resolved is not None and attrs_df.empty:
             logger.info("'attributes_encoded' is not None but resulted in an empty DataFrame (this may be valid if input was empty).")

    except Exception as e:
        logger.error(f"Error converting 'attributes_encoded' to pandas DataFrame: {e}. Input was: {ae_resolved}")
        raise TypeError(f"'attributes_encoded' could not be converted to a pandas DataFrame. Error: {e}") from e

    try:
        if ts_resolved is None:
            raise ValueError("Resolved 'timestamps' is None. A pandas Series or convertible type is required.")
        ts_series = pd.Series(ts_resolved) if not isinstance(ts_resolved, pd.Series) else ts_resolved
        if ts_series.empty:
            logger.warning("'timestamps' resolved to an empty Series.")
    except Exception as e:
        logger.error(f"Error converting 'timestamps' to pandas Series: {e}. Input was: {ts_resolved}")
        raise TypeError(f"'timestamps' could not be converted to a pandas Series. Error: {e}") from e

    # Check for index alignment - crucial for correct feature vector construction
    if not (log_series.index.equals(ts_series.index) and 
            (attrs_df.empty or log_series.index.equals(attrs_df.index))):
        logger.warning(
            "Indices of log_vectors, timestamps, and attributes_encoded (if not empty) do not match. "
            "This may lead to misaligned features or errors in FeatureExtractor. "
            f"Log/TS length: {len(log_series)}/{len(ts_series)}. Attrs length: {len(attrs_df)}. "
            f"Log index type: {type(log_series.index)}, TS index type: {type(ts_series.index)}, Attrs index type: {type(attrs_df.index) if not attrs_df.empty else 'N/A'}"
        )
        # Depending on strictness, one might raise an error here or try to align.
        # For now, proceed with warning as LogAI might handle some cases.

    cfg = FeatureExtractorConfig(max_feature_len=max_feature_len)
    extractor = FeatureExtractor(cfg)
    
    try:
        _, feat_vec = extractor.convert_to_feature_vector(log_series, attrs_df, ts_series)
    except Exception as e:
        logger.error(
            f"Error during LogAI FeatureExtractor.convert_to_feature_vector: {e}.\n"
            f"  Log series shape: {log_series.shape}, type: {type(log_series)}\n"
            f"  Attributes DF shape: {attrs_df.shape}, type: {type(attrs_df)}\n"
            f"  Timestamp series shape: {ts_series.shape}, type: {type(ts_series)}"
        )
        # Consider returning an empty DataFrame or appropriate error signal if preferred
        # For now, re-raising to make failure explicit
        raise RuntimeError(f"LogAI FeatureExtractor failed. Original error: {e}") from e

    session_vars[save_as] = feat_vec
    logger.info(f"Saved extracted log features to session_vars as '{save_as}'.")
    return feat_vec


# ---------------------------------------------------------------------------
# Counter-vector (timeseries) feature extraction
# ---------------------------------------------------------------------------


@app.tool()
@log_tool
def extract_timeseries_features(
    parsed_loglines: Any,
    attributes: Any,
    timestamps: Any,
    group_by_time: str,
    group_by_category: List[str],
    feature_extractor_params: Dict[str, Any] | None = None,
    *,
    save_as: str,
):
    """Convert logs into counter vectors (Pandas DataFrame) for time-series analysis.

    Processes parsed log lines, attributes, and timestamps to generate time-bucketed
    event counts, grouped by specified categories. The resulting Pandas DataFrame of
    counter vectors is **mandatorily** stored in `session_vars` under the key
    provided in `save_as`. This named DataFrame is then usable for time-series
    anomaly detection.

    Parameters
    ----------
    parsed_loglines : Any
        Parsed log templates or messages.
        This argument is resolved from `session_vars` if it's a string key.
        Must be a pandas Series or convertible to one.
        Example: `"parsed_templates_var"`.
    attributes : Any
        Attributes associated with each log line (e.g., hostname, application ID).
        This argument is resolved from `session_vars` if it's a string key.
        Must be a pandas DataFrame or convertible to one.
        Example: `"log_attributes_df_var"`.
    timestamps : Any
        Timestamps for each log line.
        This argument is resolved from `session_vars` if it's a string key.
        Must be a pandas Series or convertible to one.
        Example: `"log_timestamps_series_var"`.
    group_by_time : str
        The time window to group events by (e.g., "5s", "1min", "1H"). This uses
        pandas time frequency strings.
        This argument is **not** resolved from `session_vars`.
    group_by_category : List[str]
        A list of column names from the `attributes` DataFrame to use for further
        grouping the time-bucketed events. For each unique combination of these
        categories, a separate time series of counts will be generated.
        Example: `["hostname", "error_code"]`.
        This argument is **not** resolved from `session_vars`.
    feature_extractor_params : Dict[str, Any] | None, default None
        Additional parameters for LogAI's `FeatureExtractorConfig`.
        This argument is **not** resolved from `session_vars`.
    save_as : str
        The **required** key under which the resulting counter vector DataFrame
        will be stored in `session_vars`. Must be provided by the caller (LLM).

    Returns
    -------
    pandas.DataFrame
        A DataFrame where rows typically represent time windows and groups, and
        columns include counts and group identifiers. The exact structure is
        determined by LogAI's `FeatureExtractor.convert_to_counter_vector`.

    Side Effects
    ------------
    - Stores the counter vector DataFrame in `session_vars` under the key specified
      by `save_as` or an auto-generated key.
    - Raises `TypeError` if `parsed_loglines`, `attributes`, or `timestamps` are
      not of the expected pandas types after resolution.

    Examples
    --------
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
    parse_log_data : Provides `parsed_loglines`.
    preprocess_log_data : Can provide `attributes` (from its result tuple).
    load_file_log_data : Can provide `timestamps` (from `log_record.timestamp`).
    detect_timeseries_anomalies : Typically consumes the output of this tool.

    Notes
    -----
    - `parsed_loglines`, `attributes`, and `timestamps` inputs are resolved
      using `_resolve`.
    - Uses `logai.information_extraction.feature_extractor.FeatureExtractor` with
      its `convert_to_counter_vector` method.
    """

    parsed = _resolve(parsed_loglines)
    attrs = _resolve(attributes)
    ts = _resolve(timestamps)

    if not isinstance(parsed, pd.Series):
        raise TypeError("parsed_loglines must be pandas Series")
    if not isinstance(attrs, pd.DataFrame):
        raise TypeError("attributes must be pandas DataFrame")
    if not isinstance(ts, pd.Series):
        raise TypeError("timestamps must be pandas Series")

    cfg = FeatureExtractorConfig(
        group_by_time=group_by_time,
        group_by_category=group_by_category,
        **(feature_extractor_params or {}),
    )
    extractor = FeatureExtractor(cfg)
    counter_df = extractor.convert_to_counter_vector(parsed, attrs, ts)

    session_vars[save_as] = counter_df
    logger.info(f"Saved timeseries counter vector (DataFrame) to session_vars as '{save_as}'.")
    return counter_df
