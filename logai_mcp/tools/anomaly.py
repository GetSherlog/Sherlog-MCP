"""Anomaly-detection tools (time-series & semantic)."""

from typing import Any, Dict, List, Literal

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from logai_mcp.session import (
    app,
    log_tool,
    _resolve,
    session_vars,
    logger,
)

from logai.analysis.anomaly_detector import AnomalyDetector, AnomalyDetectionConfig

# ---------------------------------------------------------------------------
# Time-series anomaly detection
# ---------------------------------------------------------------------------


@app.tool()
@log_tool
def detect_timeseries_anomalies(
    counter_vector_data: Any,
    algo_name: str,
    timestamp_col: str,
    count_col: str,
    attribute_group_cols: List[str] | None = None,
    anomaly_detection_params: Dict[str, Any] | None = None,
    train_split_ratio: float = 0.7,
    *,
    save_scores_as: str,
    save_anomalies_as: str,
) -> Dict[str, Any]:
    """Detect anomalies in counter-vector time series data, saving outputs.

    Applies anomaly detection to time-series event counts (e.g., from
    `extract_timeseries_features`). Processes data, potentially grouped by attributes.
    - Anomaly scores (Pandas Series) are **mandatorily** saved to `session_vars`
      under `save_scores_as`.
    - Data points flagged as anomalous (Pandas DataFrame) are **mandatorily** saved
      to `session_vars` under `save_anomalies_as`.
    The LLM must provide names for these outputs for later analysis.

    Parameters
    ----------
    counter_vector_data : Any
        The time-series data of event counts. This argument is resolved from
        `session_vars` if it's a string key.
        Expected to be a pandas DataFrame, typically from `extract_timeseries_features`,
        containing a timestamp column, a count column, and optionally group columns.
        Example: `"host_event_counts_var"` (a key in `session_vars`).
    algo_name : str
        The name of the anomaly detection algorithm to use (e.g., "ets", "arima").
        Supported algorithms depend on LogAI's `AnomalyDetector`.
        This argument is **not** resolved from `session_vars`.
    timestamp_col : str
        The name of the column in `counter_vector_data` that contains timestamps.
        This argument is **not** resolved from `session_vars`.
    count_col : str
        The name of the column in `counter_vector_data` that contains the event counts.
        This argument is **not** resolved from `session_vars`.
    attribute_group_cols : List[str] | None, default None
        A list of column names in `counter_vector_data` to group by. Anomaly
        detection will be performed independently for each group.
        If `None` or empty, the entire dataset is treated as a single series.
        This argument is **not** resolved from `session_vars`.
    anomaly_detection_params : Dict[str, Any] | None, default None
        Algorithm-specific parameters for anomaly detection, passed to LogAI's
        `AnomalyDetectionConfig`.
        This argument is **not** resolved from `session_vars`.
    train_split_ratio : float, default 0.7
        The fraction of data to use for training the anomaly detection model for
        each time series. The model is trained on the initial part of the series
        and predicts on the latter part.
        This argument is **not** resolved from `session_vars`.
    save_scores_as : str
        **Required** key to store the Pandas Series of anomaly scores in `session_vars`.
        Must be provided by the caller (LLM).
    save_anomalies_as : str
        **Required** key to store a DataFrame of original data rows from
        `counter_vector_data` flagged as anomalous in `session_vars`.
        Must be provided by the caller (LLM).

    Returns
    -------
    Dict[str, Any]
        A dictionary summarizing the detection results:
        - "processed_series_count" (int): Number of distinct time series processed (after grouping).
        - "anomalies_found_count" (int): Total number of anomalies detected across all series.
        - "anomaly_scores_var" (str | None): The `session_vars` key for the saved scores.
        - "anomalous_data_var" (str | None): The `session_vars` key for the saved anomalous data.

    Side Effects
    ------------
    - Stores anomaly scores (pd.Series) in `session_vars`.
    - Stores anomalous data rows (pd.DataFrame) in `session_vars`.
    - Raises `TypeError` if `counter_vector_data` is not a DataFrame after resolution.

    Examples
    --------
    # Assuming session_vars["event_counts_df"] is a counter vector DataFrame:
    >>> detect_timeseries_anomalies(
    ...     counter_vector_data="event_counts_df",
    ...     algo_name="ets",
    ...     timestamp_col="time_bucket",
    ...     count_col="event_count",
    ...     attribute_group_cols=["hostname"],
    ...     save_scores_as="ts_anomaly_scores",
    ...     save_anomalies_as="ts_anomalous_events"
    ... )
    # session_vars["ts_anomaly_scores"] and session_vars["ts_anomalous_events"] will be populated.

    See Also
    --------
    extract_timeseries_features : Typically provides `counter_vector_data`.

    Notes
    -----
    - The `counter_vector_data` input is resolved using `_resolve`.
    - Data for each group is sorted by `timestamp_col` before splitting and processing.
    - Series with fewer than 10 data points are skipped.
    - LogAI's `AnomalyDetector` is used for fitting and prediction.
    - Anomaly criteria (e.g., score > 0) might be specific to LogAI's conventions.
    """

    df = _resolve(counter_vector_data)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("counter_vector_data must be DataFrame")

    ad_cfg = AnomalyDetectionConfig()
    ad_cfg.algo_name = algo_name
    ad_cfg.algo_params = anomaly_detection_params or {}
    detector = AnomalyDetector(ad_cfg)

    results = []

    group_cols = attribute_group_cols or []
    if group_cols and all(c in df.columns for c in group_cols):
        grouped = df.groupby(group_cols)
    else:
        grouped = [(None, df)]

    for key, sub in grouped:
        sub = sub.sort_values(timestamp_col)
        if len(sub) < 10:
            continue
        train_size = int(len(sub) * train_split_ratio)
        train_df = sub[[timestamp_col, count_col]].iloc[:train_size]
        test_df = sub[[timestamp_col, count_col]].iloc[train_size:]

        detector.fit(train_df)
        scores_raw = detector.predict(test_df) # Get raw prediction
        
        current_scores: pd.Series
        if isinstance(scores_raw, pd.DataFrame):
            if scores_raw.shape[1] == 1:
                current_scores = scores_raw.iloc[:, 0].rename("score")
            else:
                logger.warning(f"Timeseries AnomalyDetector.predict returned multi-column DataFrame for group {key}. Using first column.")
                current_scores = scores_raw.iloc[:, 0].rename("score")
        elif isinstance(scores_raw, np.ndarray):
            current_scores = pd.Series(scores_raw, name="score")
        elif isinstance(scores_raw, pd.Series):
            current_scores = scores_raw.rename("score") # Ensure it has a name
        else:
            logger.warning(f"Unexpected type from AnomalyDetector.predict for group {key}: {type(scores_raw)}. Converting to Series.")
            current_scores = pd.Series(scores_raw, name="score")
        
        current_scores.index = test_df.index # Set index after ensuring it's a Series
        results.append(current_scores)

    if results:
        _temp_scores_all = pd.concat(results).sort_index()
        if isinstance(_temp_scores_all, pd.DataFrame):
            if _temp_scores_all.shape[1] == 1:
                scores_all = _temp_scores_all.iloc[:, 0]
                logger.warning("Concatenated time-series scores (pd.concat(results)) unexpectedly returned a single-column DataFrame. Converted to Series.")
            else:
                logger.error(f"Concatenated time-series scores (pd.concat(results)) returned a multi-column DataFrame (shape: {_temp_scores_all.shape}). This is unexpected. Using an empty Series for scores_all.")
                scores_all = pd.Series(dtype=float) # Fallback to empty Series
        elif isinstance(_temp_scores_all, pd.Series):
            scores_all = _temp_scores_all
        else:
            logger.error(f"Concatenated time-series scores (pd.concat(results)) returned an unexpected type: {type(_temp_scores_all)}. Using an empty Series for scores_all.")
            scores_all = pd.Series(dtype=float) # Fallback
    else:
        scores_all = pd.Series(dtype=float) # Ensure scores_all is always a Series if results is empty
    
    # Ensure scores_all is numeric, coercing errors to NaN
    scores_all = pd.to_numeric(scores_all, errors='coerce')

    session_vars[save_scores_as] = scores_all
    logger.info(f"Saved timeseries anomaly scores (Series) to session_vars as '{save_scores_as}'.")

    if scores_all.empty:
        anomalies_df_to_save = pd.DataFrame()
    else:
        # Ensure index access is safe, especially if scores_all could be non-boolean
        # LogAI typically returns numerical scores; positive might indicate anomaly for some algos.
        # The condition `scores_all > 0` should be verified against LogAI's specific algo output.
        # Assuming positive scores are anomalous for this context based on original code.
        anomalous_indices = scores_all[scores_all > 0].index
        anomalies_df_to_save = df.loc[anomalous_indices] if not anomalous_indices.empty else pd.DataFrame()
    
    session_vars[save_anomalies_as] = anomalies_df_to_save
    logger.info(f"Saved timeseries anomalous data (DataFrame) to session_vars as '{save_anomalies_as}'.")

    return {
        "processed_series_count": len(results),
        "anomalies_found_count": len(anomalies_df_to_save),
        "anomaly_scores_var": save_scores_as, # Return the key used
        "anomalous_data_var": save_anomalies_as, # Return the key used
    }


# ---------------------------------------------------------------------------
# Semantic / feature-based anomaly detection
# ---------------------------------------------------------------------------


@app.tool()
@log_tool
def detect_semantic_anomalies(
    feature_vector_data: Any,
    algo_name: str,
    anomaly_detection_params: Dict[str, Any] | None = None,
    train_split_ratio: float = 0.7,
    *,
    save_predictions_as: str,
    save_anomalous_indices_as: str,
) -> Dict[str, Any]:
    """Detect semantic outliers in feature vectors, saving predictions and indices.

    Applies anomaly detection (e.g., Isolation Forest) to numerical feature vectors
    (e.g., from `extract_log_features`).
    - Prediction scores (Pandas Series) from the test set are **mandatorily** saved
      to `session_vars` under `save_predictions_as`.
    - Indices of anomalous data points (Pandas Index) are **mandatorily** saved
      to `session_vars` under `save_anomalous_indices_as`.
    The LLM must provide names for these outputs.

    Parameters
    ----------
    feature_vector_data : Any
        The numerical feature vectors to analyze for anomalies. This argument is
        resolved from `session_vars` if it's a string key.
        Expected to be a pandas DataFrame or a NumPy 2D array.
        Example: `"combined_log_features_var"` (a key in `session_vars`).
    algo_name : str
        The name of the anomaly detection algorithm (e.g., "isolation_forest",
        "lof" for Local Outlier Factor). Supported algorithms depend on LogAI's
        `AnomalyDetector`.
        This argument is **not** resolved from `session_vars`.
    anomaly_detection_params : Dict[str, Any] | None, default None
        Algorithm-specific parameters, passed to LogAI's `AnomalyDetectionConfig`.
        Example for Isolation Forest: `{"n_estimators": 100, "contamination": "auto"}`.
        This argument is **not** resolved from `session_vars`.
    train_split_ratio : float, default 0.7
        The fraction of `feature_vector_data` to use for training the anomaly
        detection model. The data is shuffled before splitting.
        This argument is **not** resolved from `session_vars`.
    save_predictions_as : str
        **Required** key to store the Pandas Series of prediction scores (on test set)
        in `session_vars`. Must be provided by the caller (LLM).
    save_anomalous_indices_as : str
        **Required** key to store a Pandas Index of anomalous data point indices
        (from test set) in `session_vars`. Must be provided by the caller (LLM).

    Returns
    -------
    Dict[str, Any]
        A dictionary summarizing the detection results:
        - "anomalies_detected_count" (int): Number of anomalies found in the test set.
        - "predictions_var" (str | None): The `session_vars` key for the saved prediction scores.
        - "anomalous_indices_var" (str | None): The `session_vars` key for the saved anomalous indices.

    Side Effects
    ------------
    - Stores prediction scores (pd.Series) in `session_vars`.
    - Stores anomalous indices (pd.Index) in `session_vars`.
    - Raises `TypeError` if `feature_vector_data` is not a DataFrame or ndarray
      after resolution.

    Examples
    --------
    # Assuming session_vars["log_fts"] is a DataFrame of features:
    >>> detect_semantic_anomalies(
    ...     feature_vector_data="log_fts",
    ...     algo_name="isolation_forest",
    ...     save_predictions_as="sem_anomaly_scores",
    ...     save_anomalous_indices_as="sem_anomalous_idx"
    ... )
    # session_vars["sem_anomaly_scores"] and session_vars["sem_anomalous_idx"] will be populated.

    See Also
    --------
    extract_log_features : Commonly provides `feature_vector_data`.
    vectorize_log_data : Can also provide `feature_vector_data`.
    quick_detect_log_anomalies : A high-level pipeline that uses similar logic.

    Notes
    -----
    - The `feature_vector_data` input is resolved using `_resolve`.
    - Data is converted to a pandas DataFrame if it's a NumPy array.
    - Uses `sklearn.model_selection.train_test_split` for data splitting (with shuffle=True).
    - LogAI's `AnomalyDetector` is used for model fitting and prediction.
    - Anomaly criteria (e.g., prediction score < 0) are based on conventions for
      algorithms like Isolation Forest.
    """

    fv = _resolve(feature_vector_data)
    if isinstance(fv, np.ndarray):
        df = pd.DataFrame(fv)
    elif isinstance(fv, pd.DataFrame):
        df = fv
    else:
        raise TypeError("feature_vector_data must be DataFrame or ndarray")

    train_df, test_df = train_test_split(df, train_size=train_split_ratio, shuffle=True)

    cfg = AnomalyDetectionConfig()
    cfg.algo_name = algo_name
    cfg.algo_params = anomaly_detection_params or {}
    det = AnomalyDetector(cfg)
    det.fit(train_df)
    raw_preds = det.predict(test_df)

    # Standardize raw_preds to be a pd.Series named 'final_preds'
    final_preds: pd.Series
    if isinstance(raw_preds, pd.Series):
        final_preds = raw_preds
    elif isinstance(raw_preds, pd.DataFrame):
        if raw_preds.shape[1] == 1:
            final_preds = raw_preds.iloc[:, 0]
        else:
            logger.warning(f"Semantic AnomalyDetector.predict returned multi-column DataFrame (shape: {raw_preds.shape}). Using first column.")
            final_preds = raw_preds.iloc[:, 0]
    elif isinstance(raw_preds, np.ndarray):
        if raw_preds.ndim == 1: # 1D array
            final_preds = pd.Series(raw_preds, index=test_df.index if hasattr(test_df, 'index') else None)
        elif raw_preds.ndim == 2 and raw_preds.shape[1] == 1: # (N,1) 2D array
            final_preds = pd.Series(raw_preds.ravel(), index=test_df.index if hasattr(test_df, 'index') else None)
        else:
            logger.error(f"Semantic AnomalyDetector.predict returned ndarray with unhandled shape: {raw_preds.shape}. Using empty Series.")
            final_preds = pd.Series(dtype=float, index=test_df.index if hasattr(test_df, 'index') else None)
    else:
        # Attempt to convert other types (e.g., list, scalar) to Series
        try:
            final_preds = pd.Series(raw_preds, index=test_df.index if hasattr(test_df, 'index') else None)
        except Exception as e:
            logger.error(f"Failed to convert raw_preds of type {type(raw_preds)} to Series: {e}. Using empty Series.")
            final_preds = pd.Series(dtype=float, index=test_df.index if hasattr(test_df, 'index') else None)
    
    preds = final_preds # Use the standardized Series

    # Ensure preds is numeric, coercing errors to NaN
    preds = pd.to_numeric(preds, errors='coerce')

    session_vars[save_predictions_as] = preds
    logger.info(f"Saved semantic anomaly predictions (Series) to session_vars as '{save_predictions_as}'.")

    # Assuming negative scores indicate anomalies for algorithms like Isolation Forest
    # Filter out NaNs that may have resulted from pd.to_numeric before comparison
    anomalous_indices = preds[preds.notna() & (preds < 0)].index
    session_vars[save_anomalous_indices_as] = anomalous_indices
    logger.info(f"Saved semantic anomalous indices (Index) to session_vars as '{save_anomalous_indices_as}'.")
    
    return {
        "anomalies_detected_count": len(anomalous_indices),
        "predictions_var": save_predictions_as, # Return the key used
        "anomalous_indices_var": save_anomalous_indices_as, # Return the key used
    }
