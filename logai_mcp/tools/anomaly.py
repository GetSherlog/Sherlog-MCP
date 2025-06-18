"""Anomaly-detection tools (time-series & semantic)
+--------------------------------------------------
This module mirrors the structure of :pyfile:`filesystem_tools.py`.

• Implementation details live in private helpers suffixed with ``_impl``.
• Public ``@app.tool`` wrappers simply delegate to those helpers by executing
  them inside the interactive IPython shell via :pyfunc:`run_code_in_shell` and
  **bind their results to names chosen by the LLM (``save_*`` args)**.

Because the actual *saving* is done by the variable assignment in the shell
statement, the helpers themselves **no longer write to ``session_vars``**.
Instead they _return_ their outputs so the wrapper can expose / save them in
the shell context.
"""

from typing import Any

import numpy as np
import pandas as pd
from logai.analysis.anomaly_detector import AnomalyDetectionConfig, AnomalyDetector
from sklearn.model_selection import train_test_split

from logai_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from logai_mcp.session import (
    app,
    logger,
)


def _detect_timeseries_anomalies_impl(
    counter_vector_data: pd.DataFrame,
    algo_name: str,
    timestamp_col: str,
    count_col: str,
    attribute_group_cols: list[str] | None = None,
    anomaly_detection_params: dict[str, Any] | None = None,
    train_split_ratio: float = 0.7,
) -> tuple[pd.Series, pd.DataFrame]:
    """Low-level implementation for detecting time-series anomalies.

    Parameters
    ----------
    counter_vector_data : pandas.DataFrame
        A DataFrame with at least the columns ``timestamp_col`` and
        ``count_col`` ‑ typically the output of
        :pyfunc:`extract_timeseries_features`.
    algo_name : str
        The anomaly-detection algorithm understood by LogAI
        (:pyclass:`logai.analysis.anomaly_detector.AnomalyDetector`).
    timestamp_col, count_col : str
        Column names inside *counter_vector_data* identifying the timestamp and
        the numeric count.
    attribute_group_cols : list[str] | None, default None
        Optional additional columns to group by.  Each group is treated as an
        independent time series.
    anomaly_detection_params : dict | None, default None
        Algo-specific keyword arguments passed to
        :pyclass:`~logai.analysis.anomaly_detector.AnomalyDetectionConfig`.
    train_split_ratio : float, default 0.7
        Fraction of each series used for training before predicting on the
        remainder.

    Returns
    -------
    tuple
        ``(scores, anomalies_df)`` where

        ``scores``
            pandas.Series of anomaly scores **indexed like the test split**.
        ``anomalies_df``
            Sub-DataFrame of *counter_vector_data* flagged as anomalous (here:
            *score* > 0).

    """
    ad_cfg = AnomalyDetectionConfig()
    ad_cfg.algo_name = algo_name
    ad_cfg.algo_params = anomaly_detection_params or {}
    detector = AnomalyDetector(ad_cfg)

    results = []

    group_cols = attribute_group_cols or []
    if group_cols and all(c in counter_vector_data.columns for c in group_cols):
        grouped = counter_vector_data.groupby(group_cols)
    else:
        grouped = [(None, counter_vector_data)]

    for key, sub in grouped:
        sub = sub.sort_values(timestamp_col)
        if len(sub) < 10:
            continue
        train_size = int(len(sub) * train_split_ratio)
        train_df = sub[[timestamp_col, count_col]].iloc[:train_size]
        test_df = sub[[timestamp_col, count_col]].iloc[train_size:]

        detector.fit(train_df)
        scores_raw = detector.predict(test_df)

        current_scores: pd.Series
        if isinstance(scores_raw, pd.DataFrame):
            if scores_raw.shape[1] == 1:
                current_scores = scores_raw.iloc[:, 0].rename("score")
            else:
                logger.warning(
                    f"Timeseries AnomalyDetector.predict returned multi-column DataFrame for group {key}. Using first column."
                )
                current_scores = scores_raw.iloc[:, 0].rename("score")
        elif isinstance(scores_raw, np.ndarray):
            current_scores = pd.Series(scores_raw, name="score")
        elif isinstance(scores_raw, pd.Series):
            current_scores = scores_raw.rename("score")
        else:
            logger.warning(
                f"Unexpected type from AnomalyDetector.predict for group {key}: {type(scores_raw)}. Converting to Series."
            )
            current_scores = pd.Series(scores_raw, name="score")

        current_scores.index = test_df.index
        results.append(current_scores)

    if results:
        _temp_scores_all = pd.concat(results).sort_index()
        if isinstance(_temp_scores_all, pd.DataFrame):
            if _temp_scores_all.shape[1] == 1:
                scores_all = _temp_scores_all.iloc[:, 0]
                logger.warning(
                    "Concatenated time-series scores (pd.concat(results)) unexpectedly returned a single-column DataFrame. Converted to Series."
                )
            else:
                logger.error(
                    f"Concatenated time-series scores (pd.concat(results)) returned a multi-column DataFrame (shape: {_temp_scores_all.shape}). This is unexpected. Using an empty Series for scores_all."
                )
                scores_all = pd.Series(dtype=float)
        elif isinstance(_temp_scores_all, pd.Series):
            scores_all = _temp_scores_all
        else:
            logger.error(
                f"Concatenated time-series scores (pd.concat(results)) returned an unexpected type: {type(_temp_scores_all)}. Using an empty Series for scores_all."
            )
            scores_all = pd.Series(dtype=float)
    else:
        scores_all = pd.Series(dtype=float)

    scores_all = pd.to_numeric(scores_all, errors="coerce")

    if scores_all.empty:
        anomalies_df = pd.DataFrame()
    else:
        anomalous_indices = scores_all[scores_all > 0].index
        anomalies_df = (
            counter_vector_data.loc[anomalous_indices]
            if not anomalous_indices.empty
            else pd.DataFrame()
        )

    return scores_all, anomalies_df


def _detect_semantic_anomalies_impl(
    feature_vector_data: Any,
    algo_name: str,
    anomaly_detection_params: dict[str, Any] | None = None,
    train_split_ratio: float = 0.7,
) -> tuple[pd.Series, pd.Index]:
    """Low-level implementation for detecting semantic / feature anomalies.

    Parameters
    ----------
    feature_vector_data : pandas.DataFrame | numpy.ndarray
        Numerical feature matrix (observations × features).
    algo_name : str
        Name of the anomaly-detection algorithm supported by LogAI.
    anomaly_detection_params : dict | None, default None
        Extra parameters for the selected algorithm.
    train_split_ratio : float, default 0.7
        Ratio of the dataset used for training.

    Returns
    -------
    tuple
        ``(predictions, anomalous_idx)`` where

        ``predictions``
            pandas.Series of prediction scores for the *test* rows.
        ``anomalous_idx``
            pandas.Index of rows whose score < 0 (Isolation-Forest style).

    """
    if isinstance(feature_vector_data, np.ndarray):
        df = pd.DataFrame(feature_vector_data)
    elif isinstance(feature_vector_data, pd.DataFrame):
        df = feature_vector_data
    else:
        raise TypeError("feature_vector_data must be DataFrame or ndarray")

    train_df, test_df = train_test_split(df, train_size=train_split_ratio, shuffle=True)

    cfg = AnomalyDetectionConfig()
    cfg.algo_name = algo_name
    cfg.algo_params = anomaly_detection_params or {}
    det = AnomalyDetector(cfg)
    det.fit(train_df)
    raw_preds = det.predict(test_df)

    final_preds: pd.Series
    if isinstance(raw_preds, pd.Series):
        final_preds = raw_preds
    elif isinstance(raw_preds, pd.DataFrame):
        if raw_preds.shape[1] == 1:
            final_preds = raw_preds.iloc[:, 0]
        else:
            logger.warning(
                f"Semantic AnomalyDetector.predict returned multi-column DataFrame (shape: {raw_preds.shape}). Using first column."
            )
            final_preds = raw_preds.iloc[:, 0]
    elif isinstance(raw_preds, np.ndarray):
        if raw_preds.ndim == 1:  # 1D array
            final_preds = pd.Series(
                raw_preds, index=test_df.index if hasattr(test_df, "index") else None
            )
        elif raw_preds.ndim == 2 and raw_preds.shape[1] == 1:  # (N,1) 2D array
            final_preds = pd.Series(
                raw_preds.ravel(),
                index=test_df.index if hasattr(test_df, "index") else None,
            )
        else:
            logger.error(
                f"Semantic AnomalyDetector.predict returned ndarray with unhandled shape: {raw_preds.shape}. Using empty Series."
            )
            final_preds = pd.Series(
                dtype=float, index=test_df.index if hasattr(test_df, "index") else None
            )
    else:
        try:
            final_preds = pd.Series(
                raw_preds, index=test_df.index if hasattr(test_df, "index") else None
            )
        except Exception as e:
            logger.error(
                f"Failed to convert raw_preds of type {type(raw_preds)} to Series: {e}. Using empty Series."
            )
            final_preds = pd.Series(
                dtype=float, index=test_df.index if hasattr(test_df, "index") else None
            )

    preds = final_preds

    preds = pd.to_numeric(preds, errors="coerce")

    anomalous_indices = preds[preds.notna() & (preds < 0)].index

    return preds, anomalous_indices


_SHELL.push(
    {
        "_detect_timeseries_anomalies_impl": _detect_timeseries_anomalies_impl,
        "_detect_semantic_anomalies_impl": _detect_semantic_anomalies_impl,
    }
)


@app.tool()
async def detect_timeseries_anomalies(
    counter_vector_data: Any,
    algo_name: str,
    timestamp_col: str,
    count_col: str,
    attribute_group_cols: list[str] | None = None,
    anomaly_detection_params: dict[str, Any] | None = None,
    train_split_ratio: float = 0.7,
    *,
    save_scores_as: str,
    save_anomalies_as: str,
):
    code = (
        f"{save_scores_as}, {save_anomalies_as} = _detect_timeseries_anomalies_impl("
        f"    {repr(counter_vector_data)}, "
        f"    {repr(algo_name)}, "
        f"    {repr(timestamp_col)}, "
        f"    {repr(count_col)}, "
        f"    {repr(attribute_group_cols)}, "
        f"    {repr(anomaly_detection_params)}, "
        f"    {train_split_ratio})\n"
        f"({save_scores_as}, {save_anomalies_as})"
    )
    return await run_code_in_shell(code)


detect_timeseries_anomalies.__doc__ = _detect_timeseries_anomalies_impl.__doc__


@app.tool()
async def detect_semantic_anomalies(
    feature_vector_data: Any,
    algo_name: str,
    anomaly_detection_params: dict[str, Any] | None = None,
    train_split_ratio: float = 0.7,
    *,
    save_predictions_as: str,
    save_anomalous_indices_as: str,
):
    code = (
        f"{save_predictions_as}, {save_anomalous_indices_as} = _detect_semantic_anomalies_impl("
        f"    {repr(feature_vector_data)}, "
        f"    {repr(algo_name)}, "
        f"    {repr(anomaly_detection_params)}, "
        f"    {train_split_ratio})\n"
        f"({save_predictions_as}, {save_anomalous_indices_as})"
    )
    return await run_code_in_shell(code)


detect_semantic_anomalies.__doc__ = _detect_semantic_anomalies_impl.__doc__
