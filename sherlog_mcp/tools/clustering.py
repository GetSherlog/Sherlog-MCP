"""Clustering utilities (k-means etc.).

Follows the same execution model as `filesystem_tools.py`:

• Heavy logic lives in a private `_cluster_log_features_impl` helper.
• The public `cluster_log_features` wrapper merely assigns its return value
  to the caller-supplied `save_as` variable inside the IPython shell.
"""

from typing import Any

import numpy as np
import pandas as pd
from logai.algorithms.clustering_algo.kmeans import KMeansParams
from logai.analysis.clustering import Clustering, ClusteringConfig

from sherlog_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from sherlog_mcp.session import (
    app,
    logger,
)


def _cluster_log_features_impl(
    feature_vector: Any,
    algo_name: str = "kmeans",
    n_clusters: int = 7,
    clustering_params: dict | None = None,
) -> pd.Series:
    """Group log entries by features using clustering, returning a Pandas Series of cluster IDs.

    Applies a clustering algorithm (e.g., k-means) to numerical log features
    (e.g., from `extract_log_features`). The resulting Pandas Series, mapping
    log entry indices to cluster IDs, is **mandatorily** stored in `session_vars`
    under the key provided in `save_as`. This allows the LLM to use the named
    cluster assignments for analysis.

    Parameters
    ----------
    feature_vector : Any
        The numerical feature vectors to be clustered. This argument is resolved
        from `session_vars` if it's a string key.
        Expected to be a pandas DataFrame or a NumPy 2D array.
        Example: `"combined_log_features_var"` (a key in `session_vars`).
    algo_name : str, default "kmeans"
        The name of the clustering algorithm to use. LogAI's `Clustering` class
        supports various algorithms (e.g., "kmeans", "dbscan").
        This argument is **not** resolved from `session_vars`.
    n_clusters : int, default 7
        The number of clusters to form. This is a primary parameter for
        algorithms like k-means.
        This argument is **not** resolved from `session_vars`.
    clustering_params : Dict | None, default None
        A dictionary of additional parameters specific to the chosen `algo_name`.
        These are passed to the algorithm's parameter class (e.g., `KMeansParams`
        for "kmeans") or directly to `ClusteringConfig`.
        If `algo_name` is "kmeans", `n_clusters` from the main arguments is
        prioritized, but other `KMeansParams` can be set here (e.g., `{"algorithm": "elkan"}`).
        This argument is **not** resolved from `session_vars`.

    Returns
    -------
    pandas.Series
        A Series where the index aligns with the input `feature_vector`'s rows,
        and values are the assigned cluster IDs (as strings).
        The Series is named "cluster_id".

    Side Effects
    ------------
    - Stores the pandas Series of cluster IDs in `session_vars` under the key
      specified by `save_as` or an auto-generated key.
    - Raises `TypeError` if `feature_vector` is not a DataFrame or NumPy array
      after resolution.

    Examples
    --------
    # Assuming session_vars["features_for_clustering"] is a DataFrame or NumPy array:
    >>> cluster_log_features(
    ...     feature_vector="features_for_clustering",
    ...     algo_name="kmeans",
    ...     n_clusters=5,
    ...     save_as="log_clusters_kmeans5"
    ... )
    # session_vars["log_clusters_kmeans5"] will store the Series of cluster assignments.

    See Also
    --------
    extract_log_features : Commonly provides the `feature_vector` input.
    vectorize_log_data : Can also provide `feature_vector` if attributes are not used.
    quick_cluster_logs : A high-level pipeline that uses this tool.

    Notes
    -----
    - The `feature_vector` input is resolved using `_resolve`.
    - Input is converted to a pandas DataFrame if it's a NumPy array.
    - Uses LogAI's `Clustering` and `ClusteringConfig`. For "kmeans",
      `KMeansParams` is used, incorporating `n_clusters`.
    - Cluster IDs are returned as strings in the Series.
    
    Data parameters can be DataFrame variables from previous tool calls.
    Results persist as save_as. Use execute_python_code("{save_as}.value_counts()") to see cluster sizes.

    """
    if isinstance(feature_vector, pd.DataFrame):
        df = feature_vector
    elif isinstance(feature_vector, np.ndarray):
        df = pd.DataFrame(feature_vector)
    else:
        raise TypeError("feature_vector must be DataFrame or numpy array")

    clustering_params = clustering_params or {}
    if algo_name == "kmeans":
        algo_params_obj = KMeansParams()
        algo_params_obj.n_clusters = n_clusters
        algo_params_obj.algorithm = (
            "lloyd"
        )
        for key, value in clustering_params.items():
            if hasattr(algo_params_obj, key):
                setattr(algo_params_obj, key, value)
            else:
                logger.warning(
                    f"KMeansParams does not have attribute '{key}'. It will be ignored."
                )
        algo_params = algo_params_obj
    else:
        algo_params = clustering_params  # type: ignore[assignment]

    cfg = ClusteringConfig()
    cfg.algo_name = algo_name
    cfg.algo_params = algo_params
    clusterer = Clustering(cfg)
    clusterer.fit(df)
    ids = clusterer.predict(df)
    series = pd.Series(
        ids.astype(str), name="cluster_id", index=df.index
    )

    return series


_SHELL.push({"_cluster_log_features_impl": _cluster_log_features_impl})


@app.tool()
async def cluster_log_features(
    feature_vector: Any,
    algo_name: str = "kmeans",
    n_clusters: int = 7,
    clustering_params: dict | None = None,
    *,
    save_as: str,
):
    """Execute clustering inside the IPython shell and bind its result.

    The heavy lifting is done by `_cluster_log_features_impl` (see its
    docstring for full parameter semantics).  This wrapper simply assigns the
    returned `pd.Series` to the variable name provided via `save_as` **inside
    the interactive shell** so that subsequent tool calls (or the user) can
    reference it by name.

    Parameters
    ----------
    save_as : str
        Name of the Python variable that will hold the resulting Series inside
        the shell environment.

    """
    code = (
        f"{save_as} = _cluster_log_features_impl("  # assignment\n"
        f"    {repr(feature_vector)}, {repr(algo_name)}, {n_clusters}, {repr(clustering_params)})\n"
        f"{save_as}"
    )
    execution_result = await run_code_in_shell(code)
    return execution_result.result if execution_result else None


cluster_log_features.__doc__ = _cluster_log_features_impl.__doc__
