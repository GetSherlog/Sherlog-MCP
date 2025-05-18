"""High-level, opinionated one-shot helpers.

These *pipeline* functions string together the individual low-level
building-blocks (load → preprocess → parse → …) and expose a *single*
call that takes a path to a CSV-style log file and immediately returns
• either the rows flagged as anomalous or
• the cluster assignment for every record.

Both helpers try to *do-the-right-thing™* out-of-the-box while still
exposing a handful of knobs for advanced tweaking.
"""

from typing import Any, Dict, List, Sequence
import os

import pandas as pd
import numpy as np

from logai_mcp.session import app, log_tool, session_vars, logger

# Re-use the existing *atomic* tools to keep the implementation minimal
from logai_mcp.tools.data_loading import load_file_log_data, suggest_dimension_mapping
from logai_mcp.tools.preprocessing import preprocess_log_data, parse_log_data
from logai_mcp.tools.vectorization import vectorize_log_data, encode_log_attributes
from logai_mcp.tools.feature_extraction import extract_log_features
from logai_mcp.tools.clustering import cluster_log_features

from logai.analysis.anomaly_detector import AnomalyDetector, AnomalyDetectionConfig

# ---------------------------------------------------------------------------
# Helper utilities (internal)
# ---------------------------------------------------------------------------

def _default_regex_replacements() -> List[List[str]]:
    """Return a curated list of generic patterns → placeholders.

    The list purposefully stays *very* small to keep the risk of over-
    matching low.  Users can pass their own ``custom_replace_list`` to
    override / extend the defaults.
    """
    return [
        # IPv4 addresses → <IP>
        [r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<IP>"],
        # UUID-like hex strings → <UUID>
        [r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", "<UUID>"],
        # Plain integers (long sequences only) → <NUM>
        [r"\b\d{3,}\b", "<NUM>"],
    ]


def _first_series(df: pd.DataFrame) -> pd.Series:
    """Return the *first* column of *df* as a Series.

    This is handy for LogAI data-objects where we commonly need a plain
    timestamp Series but the library gives us a single-column DataFrame.
    """
    if df.empty:
        raise ValueError("Expected non-empty DataFrame while extracting column.")
    return df.iloc[:, 0]


# ---------------------------------------------------------------------------
# End-to-end anomaly detection
# ---------------------------------------------------------------------------


@app.tool()
@log_tool
def quick_detect_log_anomalies(
    file_path: str,
    *,
    dimensions: Dict[str, Sequence[str]] | None = None,
    custom_replace_list: List[List[str]] | None = None,
    datetime_format: str | None = None,
    # IE & feature params
    vectorizer_algo: str = "word2vec",
    encoder_name: str = "label_encoder",
    feature_max_len: int = 100,
    # Anomaly detector params
    anomaly_algo: str = "isolation_forest",
    anomaly_algo_params: Dict[str, Any] | None = None,
    train_split_ratio: float = 0.7,
    # Output handling
    save_anomalies_as: str,
    save_predictions_as: str,
):
    """Run an end-to-end log anomaly detection pipeline, saving key outputs.

    Orchestrates loading, preprocessing, parsing, vectorization, feature extraction,
    and anomaly detection from a log file.
    - A DataFrame of log rows flagged as anomalous is **mandatorily** saved to
      `session_vars` under `save_anomalies_as`.
    - A Pandas Series of anomaly prediction scores for all rows is **mandatorily**
      saved to `session_vars` under `save_predictions_as`.
    The LLM must provide names for these DataFrame/Series outputs for analysis.

    Parameters
    ----------
    file_path : str
        Path to the log file (CSV / JSON-lines style). This argument is **not**
        resolved from `session_vars`; it must be a direct path string.
    dimensions : Dict[str, Sequence[str]] | None, default None
        Explicit mapping for column roles (e.g., {"timestamp": ["col_A"], "body": ["col_B"]}).
        If `None`, `suggest_dimension_mapping` is invoked internally using `file_path`.
        This argument is **not** resolved from `session_vars`.
    custom_replace_list : List[List[str]] | None, default None
        Regex replacements for preprocessing (e.g., `[["IP_REGEX", "<IP>"]]`).
        If `None`, a default list (`_default_regex_replacements`) targeting common
        patterns like IPs, UUIDs, and long numbers is used.
        This argument is **not** resolved from `session_vars`.
    datetime_format : str | None, default None
        The string format for parsing datetime strings in the log file (e.g., `"%Y-%m-%d %H:%M:%S"`
        or `[%d/%b/%Y:%H:%M:%S]`). If provided, this format is used directly, and
        datetime inference is disabled. If `None`, datetime formats are inferred.
        This argument is **not** resolved from `session_vars`.
    vectorizer_algo : str, default "word2vec"
        Algorithm for converting parsed log templates to vectors (e.g., "tfidf", "fasttext").
        Passed to `vectorize_log_data`.
        This argument is **not** resolved from `session_vars`.
    encoder_name : str, default "label_encoder"
        Algorithm for encoding categorical attributes (e.g., "one_hot_encoder").
        Passed to `encode_log_attributes`.
        This argument is **not** resolved from `session_vars`.
    feature_max_len : int, default 100
        Maximum length for combined features. Passed to `extract_log_features`.
        This argument is **not** resolved from `session_vars`.
    anomaly_algo : str, default "isolation_forest"
        Anomaly detection algorithm (e.g., "lof"). Passed to `AnomalyDetector`.
        This argument is **not** resolved from `session_vars`.
    anomaly_algo_params : Dict[str, Any] | None, default None
        Parameters for the `anomaly_algo`. Passed to `AnomalyDetectorConfig`.
        This argument is **not** resolved from `session_vars`.
    train_split_ratio : float, default 0.7
        Fraction of data used to train the anomaly detector. The detector is fit
        on the first part and predicts on the entire dataset.
        This argument is **not** resolved from `session_vars`.
    save_anomalies_as : str
        **Required** key to save the DataFrame of anomalous log rows in `session_vars`.
        Must be provided by the caller (LLM).
    save_predictions_as : str
        **Required** key to save the Pandas Series of anomaly prediction scores
        (for all rows) in `session_vars`. Must be provided by the caller (LLM).

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the original log rows that were flagged as anomalous,
        augmented with an "anomaly_score" column. Scores < 0 typically indicate outliers.

    Side Effects
    ------------
    - Invokes multiple tools (`load_file_log_data`, `preprocess_log_data`, etc.),
      each of which may save intermediate results to `session_vars` using their
      own auto-naming logic if their respective `save_*_as` parameters are not set
      by this pipeline (this pipeline typically lets them auto-save).
    - Saves the final anomaly prediction scores (pd.Series) to `session_vars` under
      `save_predictions_as` or an auto-generated key.
    - Saves the DataFrame of anomalous rows to `session_vars` under `save_anomalies_as`
      or an auto-generated key.

    Workflow (internal calls)
    -------------------------
    1. `suggest_dimension_mapping` (if `dimensions` is None)
    2. `load_file_log_data`
    3. `preprocess_log_data`
    4. `parse_log_data`
    5. `vectorize_log_data`
    6. `encode_log_attributes`
    7. `extract_log_features`
    8. `AnomalyDetector.fit()` (on training split of features)
    9. `AnomalyDetector.predict()` (on all features)

    Examples
    --------
    >>> quick_detect_log_anomalies(
    ...     file_path="./logs/server_access.csv",
    ...     vectorizer_algo="tfidf",
    ...     save_anomalies_as="server_anomalies",
    ...     save_predictions_as="server_anomaly_scores"
    ... )
    # Anomalous rows from server_access.csv are returned and saved in
    # session_vars["server_anomalies"].
    # All prediction scores are saved in session_vars["server_anomaly_scores"].

    Notes
    -----
    - This is a high-level, opinionated pipeline. For more granular control,
      use the individual tools it wraps.
    - The `file_path` should point to a CSV or JSON-lines file that LogAI can parse.
    - Default regex replacements include common patterns like IPs, UUIDs, and numbers.
    - If `datetime_format` is not provided, datetime parsing will be attempted via inference
      by the underlying `load_file_log_data` tool. If `datetime_format` is provided
      (e.g., `"%Y-%m-%d %H:%M:%S"` or `[%d/%b/%Y:%H:%M:%S]`), it will be used for
      parsing timestamps, and datetime inference will be disabled.
    """
    # ------------------------------------------------------------------
    # Step 1/6: Load file
    # ------------------------------------------------------------------
    # Generate unique internal names for intermediate results
    _pipeline_run_id = f"pipe_anom_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}"
    _dim_map_key = f"{_pipeline_run_id}_dim_map"
    _log_record_key = f"{_pipeline_run_id}_log_record"
    _clean_logs_key = f"{_pipeline_run_id}_clean_logs"
    _attributes_key = f"{_pipeline_run_id}_attributes"
    _parsed_templates_key = f"{_pipeline_run_id}_parsed_templates"
    _log_vectors_key = f"{_pipeline_run_id}_log_vectors"
    _attrs_encoded_key = f"{_pipeline_run_id}_attrs_encoded"
    _feature_vec_key = f"{_pipeline_run_id}_feature_vec"

    dim_map = dimensions
    if not dim_map:
        logger.info(f"No dimensions provided for quick_detect_log_anomalies. Suggesting and saving to '{_dim_map_key}'.")
        dim_map = suggest_dimension_mapping(file_path, save_as=_dim_map_key)
    
    _infer_dt = True
    if datetime_format:
        _infer_dt = False
    log_record = load_file_log_data(
        file_path,
        dim_map, # type: ignore
        datetime_format=datetime_format, # type: ignore
        infer_datetime=_infer_dt,
        save_as=_log_record_key
    )

    # ------------------------------------------------------------------
    # Step 2/6: Preprocess (regex replacements)
    # ------------------------------------------------------------------
    repl = custom_replace_list or _default_regex_replacements()
    # preprocess_log_data returns a tuple, we need the components
    clean_logs, _patterns, attributes = preprocess_log_data(
        _log_record_key, # Pass the key to resolve LogRecordObject
        repl,
        save_clean_as=_clean_logs_key,
        save_attributes_as=_attributes_key
        # save_patterns_as and save_as (for tuple) are optional and not strictly needed internally by pipeline
    )

    # ------------------------------------------------------------------
    # Step 3/6: Parsing → templates
    # ------------------------------------------------------------------
    parsed_templates = parse_log_data(_clean_logs_key, save_as=_parsed_templates_key)

    # ------------------------------------------------------------------
    # Step 4/6: Vectorise + encode categorical attributes
    # ------------------------------------------------------------------
    log_vectors = vectorize_log_data(_parsed_templates_key, algo_name=vectorizer_algo, save_as=_log_vectors_key)
    attrs_encoded = encode_log_attributes(_attributes_key, encoder_name=encoder_name, save_as=_attrs_encoded_key)

    # ------------------------------------------------------------------
    # Step 5/6: Concatenate features (semantic + structured)
    # ------------------------------------------------------------------
    # log_record is already resolved and available. We need its timestamp attribute.
    # Ensure log_record is the actual object, not the key, if used directly.
    # Or, better, if timestamps are saved by load_file_log_data, resolve that.
    # For now, assuming log_record object is in memory from the call above.
    ts_series = _first_series(log_record.timestamp) # type: ignore[attr-defined]
    
    feature_vec = extract_log_features(
        _log_vectors_key, _attrs_encoded_key, ts_series, # ts_series needs to be from resolved log_record or saved separately
        max_feature_len=feature_max_len,
        save_as=_feature_vec_key
    )
    if isinstance(feature_vec, np.ndarray):
        feature_df = pd.DataFrame(feature_vec)
    else:
        feature_df = feature_vec

    # ------------------------------------------------------------------
    # Step 6/6: Anomaly detection
    # ------------------------------------------------------------------
    cfg = AnomalyDetectionConfig()
    cfg.algo_name = anomaly_algo
    cfg.algo_params = anomaly_algo_params or {}
    det = AnomalyDetector(cfg)

    # *Fit* on the first ``train_split_ratio`` fraction to mimic a typical
    # production setting where we only see *past* data during training.
    train_size = int(len(feature_df) * train_split_ratio)
    det.fit(feature_df.iloc[:train_size])

    # *Predict* on the *entire* set so each record has a score.
    preds = det.predict(feature_df)
    if isinstance(preds, pd.DataFrame) and preds.shape[1] == 1:
        preds = preds.iloc[:, 0]
    elif isinstance(preds, np.ndarray):
        preds = pd.Series(preds, index=feature_df.index)

    preds.name = "anomaly_score"

    # Assemble final DataFrame
    # log_record is the LogRecordObject from load_file_log_data
    # preds is the Series of anomaly scores with name "anomaly_score"

    # Start with a DataFrame based on preds' index for alignment.
    # This ensures that even if other components are missing,
    # the anomaly scores can be presented with their correct index.
    rows_df = pd.DataFrame(index=preds.index)

    # 1. Add timestamp from the original log_record
    # log_record is the variable holding the result of load_file_log_data
    if log_record.timestamp is not None and not log_record.timestamp.empty:
        # log_record.timestamp is a DataFrame, typically with one column.
        # Use its actual column name and then rename to "timestamp" for standardization.
        ts_col_name = log_record.timestamp.columns[0]
        ts_s = log_record.timestamp[ts_col_name].rename("timestamp")
        rows_df = rows_df.join(ts_s, how="left") # Use left join to preserve all rows_df indices

    # 2. Add the cleaned log body
    # _clean_logs_key stores the result of preprocess_log_data's cleaned logs
    clean_logs_df = session_vars.get(_clean_logs_key)  # This is a DataFrame
    if clean_logs_df is not None and not clean_logs_df.empty:
        # Assuming the first column of clean_logs_df is the body.
        # Rename it to 'body' for standardized output.
        body_s = clean_logs_df.iloc[:, 0].rename("body")
        rows_df = rows_df.join(body_s, how="left")

    # 3. Add attributes extracted during preprocessing
    # _attributes_key stores the attributes DataFrame from preprocess_log_data
    attributes_df = session_vars.get(_attributes_key)  # This is a DataFrame
    if attributes_df is not None and not attributes_df.empty:
        # Prefix attribute columns to make their origin clear and avoid name clashes.
        renamed_attributes_df = attributes_df.rename(columns=lambda c: f"attr_{c}")
        rows_df = rows_df.join(renamed_attributes_df, how="left")
    
    # Now, join the constructed rows_df with the anomaly prediction scores
    result_df = rows_df.join(preds) # preds is a Series named "anomaly_score"
    
    # Filter for anomalous rows
    anomalies_df = result_df[result_df["anomaly_score"] < 0].copy()

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------
    session_vars[save_predictions_as] = preds
    logger.info(f"Saved anomaly predictions (Series) to session_vars as '{save_predictions_as}'.")

    session_vars[save_anomalies_as] = anomalies_df
    logger.info(f"Saved anomalous data (DataFrame) to session_vars as '{save_anomalies_as}'.")

    return anomalies_df


# ---------------------------------------------------------------------------
# End-to-end clustering
# ---------------------------------------------------------------------------


@app.tool()
@log_tool
def quick_cluster_logs(
    file_path: str,
    *,
    dimensions: Dict[str, Sequence[str]] | None = None,
    custom_replace_list: List[List[str]] | None = None,
    datetime_format: str | None = None,
    # IE & feature params
    vectorizer_algo: str = "word2vec",
    encoder_name: str = "label_encoder",
    feature_max_len: int = 100,
    # Clustering params
    clustering_algo: str = "kmeans",
    n_clusters: int = 7,
    clustering_params: Dict[str, Any] | None = None,
    # Output handling
    save_cluster_ids_as: str,
):
    """Run an end-to-end log clustering pipeline, saving cluster ID assignments.

    Orchestrates loading, preprocessing, parsing, vectorization, feature extraction,
    and clustering from a log file. The primary output, a Pandas Series mapping
    original log entry indices to cluster IDs, is **mandatorily** stored in
    `session_vars` under the key provided in `save_cluster_ids_as`.
    The LLM must provide a name for this output Series.

    Parameters
    ----------
    file_path : str
        Path to the log file (CSV / JSON-lines style). This argument is **not**
        resolved from `session_vars`; it must be a direct path string.
    dimensions : Dict[str, Sequence[str]] | None, default None
        Explicit mapping for column roles (e.g., {"timestamp": ["col_A"], "body": ["col_B"]}).
        If `None`, `suggest_dimension_mapping` is invoked internally using `file_path`.
        This argument is **not** resolved from `session_vars`.
    custom_replace_list : List[List[str]] | None, default None
        Regex replacements for preprocessing (e.g., `[["IP_REGEX", "<IP>"]]`).
        If `None`, a default list (`_default_regex_replacements`) targeting common
        patterns like IPs, UUIDs, and long numbers is used.
        This argument is **not** resolved from `session_vars`.
    datetime_format : str | None, default None
        The string format for parsing datetime strings in the log file (e.g., `"%Y-%m-%d %H:%M:%S"`
        or `[%d/%b/%Y:%H:%M:%S]`). If provided, this format is used directly, and
        datetime inference is disabled. If `None`, datetime formats are inferred based on
        the defaults of `load_file_log_data`.
        This argument is **not** resolved from `session_vars`.
    vectorizer_algo : str, default "word2vec"
        Algorithm for converting parsed log templates to vectors (e.g., "tfidf", "fasttext").
        Passed to `vectorize_log_data`.
        This argument is **not** resolved from `session_vars`.
    encoder_name : str, default "label_encoder"
        Algorithm for encoding categorical attributes (e.g., "one_hot_encoder").
        Passed to `encode_log_attributes`.
        This argument is **not** resolved from `session_vars`.
    feature_max_len : int, default 100
        Maximum length for combined features. Passed to `extract_log_features`.
        This argument is **not** resolved from `session_vars`.
    clustering_algo : str, default "kmeans"
        Clustering algorithm (e.g., "dbscan"). Passed to `cluster_log_features`.
        This argument is **not** resolved from `session_vars`.
    n_clusters : int, default 7
        Target number of clusters, primarily for "kmeans". Passed to `cluster_log_features`.
        This argument is **not** resolved from `session_vars`.
    clustering_params : Dict[str, Any] | None, default None
        Parameters for the `clustering_algo`. Passed to `cluster_log_features`.
        This argument is **not** resolved from `session_vars`.
    save_cluster_ids_as : str
        **Required** key to save the Pandas Series of cluster IDs in `session_vars`.
        Must be provided by the caller (LLM).

    Returns
    -------
    pandas.Series
        A Series where the index aligns with the original log entries and values are
        the assigned cluster IDs (as strings).

    Side Effects
    ------------
    - Invokes multiple tools (`load_file_log_data`, `preprocess_log_data`, etc.),
      each of which may save intermediate results to `session_vars` using their
      own auto-naming logic if their respective `save_*_as` parameters are not set
      by this pipeline (this pipeline typically lets them auto-save).
    - Calls `cluster_log_features` which saves its result (the cluster ID Series)
      to `session_vars` under the `save_clusters_as` parameter of that function (this
      pipeline does *not* pass a `save_clusters_as` to the underlying call, so
      `cluster_log_features` will use its own auto-naming for that internal step).
    - This `quick_cluster_logs` function then explicitly saves the same cluster ID Series
      again into `session_vars` under the key specified by its own `save_cluster_ids_as`
      parameter (or an auto-generated one if `save_cluster_ids_as` is None).

    Workflow (internal calls)
    -------------------------
    1. `suggest_dimension_mapping` (if `dimensions` is None)
    2. `load_file_log_data`
    3. `preprocess_log_data`
    4. `parse_log_data`
    5. `vectorize_log_data`
    6. `encode_log_attributes`
    7. `extract_log_features`
    8. `cluster_log_features`

    Examples
    --------
    >>> quick_cluster_logs(
    ...     file_path="./logs/application.log.csv",
    ...     n_clusters=10,
    ...     save_cluster_ids_as="app_log_clusters"
    ... )
    # Cluster IDs for application.log.csv entries are returned and saved in
    # session_vars["app_log_clusters"].

    Notes
    -----
    - This is a high-level, opinionated pipeline. For more granular control,
      use the individual tools it wraps.
    - The `file_path` should point to a CSV or JSON-lines file.
    - If `datetime_format` is not provided, datetime parsing relies on the default
      behavior of `load_file_log_data` (which might include inference).
      If `datetime_format` is provided (e.g., `"%Y-%m-%d %H:%M:%S"` or
      `[%d/%b/%Y:%H:%M:%S]`), it will be used for parsing timestamps, and
      datetime inference will be disabled.
    """
    # ------------------------------------------------------------------
    # Load & preprocess identical to anomaly pipeline
    # ------------------------------------------------------------------
    _pipeline_run_id = f"pipe_clust_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}"
    _dim_map_key = f"{_pipeline_run_id}_dim_map"
    _log_record_key = f"{_pipeline_run_id}_log_record"
    _clean_logs_key = f"{_pipeline_run_id}_clean_logs"
    _attributes_key = f"{_pipeline_run_id}_attributes"
    _parsed_templates_key = f"{_pipeline_run_id}_parsed_templates"
    _log_vectors_key = f"{_pipeline_run_id}_log_vectors"
    _attrs_encoded_key = f"{_pipeline_run_id}_attrs_encoded"
    _feature_vec_key = f"{_pipeline_run_id}_feature_vec" # Key for features before clustering

    dim_map = dimensions
    if not dim_map:
        logger.info(f"No dimensions provided for quick_cluster_logs. Suggesting and saving to '{_dim_map_key}'.")
        dim_map = suggest_dimension_mapping(file_path, save_as=_dim_map_key)

    _infer_dt = True
    if datetime_format:
        _infer_dt = False
    log_record = load_file_log_data(
        file_path,
        dim_map, # type: ignore
        datetime_format=datetime_format, # type: ignore
        infer_datetime=_infer_dt,
        save_as=_log_record_key
    )

    repl = custom_replace_list or _default_regex_replacements()
    # preprocess_log_data returns a tuple
    clean_logs, _patterns, attributes = preprocess_log_data(
        _log_record_key, # Pass key
        repl,
        save_clean_as=_clean_logs_key,
        save_attributes_as=_attributes_key
    )
    parsed_templates = parse_log_data(_clean_logs_key, save_as=_parsed_templates_key)

    log_vectors = vectorize_log_data(_parsed_templates_key, algo_name=vectorizer_algo, save_as=_log_vectors_key)
    attrs_encoded = encode_log_attributes(_attributes_key, encoder_name=encoder_name, save_as=_attrs_encoded_key)
    
    # Resolve log_record for timestamp or ensure timestamps are saved and resolved by key
    # For now, using the in-memory log_record from above.
    ts_series = _first_series(log_record.timestamp)  # type: ignore[attr-defined] 
    
    _feature_data = extract_log_features(
        _log_vectors_key, _attrs_encoded_key, ts_series, # ts_series needs to be from resolved log_record or saved separately
        max_feature_len=feature_max_len,
        save_as=_feature_vec_key # Save the features before clustering
    )
    if isinstance(_feature_data, np.ndarray):
        feature_df_for_clustering = pd.DataFrame(_feature_data)
    else:
        feature_df_for_clustering = _feature_data # Assume it's already a DataFrame

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    # cluster_log_features itself now requires save_as.
    # The pipeline's save_cluster_ids_as is for the *final* output of this pipeline.
    # So, cluster_log_features will save its output under the name provided by this pipeline's save_cluster_ids_as.
    cluster_results_series = cluster_log_features(
        _feature_vec_key, # Pass the key for the feature data
        clustering_algo=clustering_algo,
        n_clusters=n_clusters,
        clustering_params=clustering_params,
        save_as=save_cluster_ids_as # Use the pipeline's designated save_as for the final cluster IDs
    )
    # No need to save again here, as cluster_log_features already saved it under save_cluster_ids_as
    logger.info(f"Cluster IDs (Series) were saved by cluster_log_features to session_vars as '{save_cluster_ids_as}'.")
    return cluster_results_series
