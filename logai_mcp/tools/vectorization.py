"""Vectorisation & attribute encoding tools.

Refactored to follow the `filesystem_tools.py` pattern:
    • Heavy logic moved to private helpers suffixed with `_impl`.
    • Public `@app.tool` wrappers execute those helpers inside the IPython shell
      via `run_code_in_shell`.
    • Helpers are exposed to the shell through `_SHELL.push(...)`.
"""

from typing import Any

import numpy as np
import pandas as pd
from logai.information_extraction.categorical_encoder import (
    CategoricalEncoder,
    CategoricalEncoderConfig,
)
from logai.information_extraction.log_vectorizer import LogVectorizer, VectorizerConfig

from logai_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from logai_mcp.session import (
    app,
)

# ---------------------------------------------------------------------------
# Vectorise parsed log templates
# ---------------------------------------------------------------------------


def _vectorize_log_data_impl(
    parsed_loglines: Any,
    algo_name: str = "word2vec",
    vectorizer_params: dict | None = None,
) -> np.ndarray:
    """Convert textual log data into numerical feature vectors (NumPy array).

    Transforms parsed log templates or messages (e.g., from `parse_log_data`)
    into numerical vectors using algorithms like "word2vec", "tfidf", etc.
    The resulting NumPy array of feature vectors is **mandatorily** stored in
    `session_vars` under the key provided in `save_as`. This named output
    allows the LLM to use these vectors for subsequent ML tasks.

    Parameters
    ----------
    parsed_loglines : Any
        The textual log data (e.g., templates, messages) to be vectorized.
        This argument is resolved from `session_vars` if it's a string key. It can be:
        - A pandas Series of template strings.
        - A pandas DataFrame where the first column contains template strings.
        - A list or tuple of template strings.
        - A NumPy array of template strings.
        Example: `"log_templates_var"` (a key in `session_vars`).
    algo_name : str, default "word2vec"
        The name of the vectorization algorithm to use. Supported algorithms
        depend on the LogAI library's `LogVectorizer`.
        Common options: "word2vec", "tfidf", "fasttext", "forecast_bert".
        This argument is **not** resolved from `session_vars`.
    vectorizer_params : Dict | None, default None
        Optional dictionary of parameters specific to the chosen `algo_name`.
        These are passed to LogAI's `VectorizerConfig`.
        Example for "word2vec": `{"model_params": {"vector_size": 100, "window": 5}}`.
        This argument is **not** resolved from `session_vars`.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array where each row is a numerical feature vector representing a
        log template or line. The shape will be (number_of_loglines, feature_dimension).

    Side Effects
    ------------
    - Stores the NumPy array of numerical feature vectors in `session_vars` under
      the key specified by `save_as` or an auto-generated key.
    - Raises `TypeError` if `parsed_loglines` is not of a supported type (list,
      Series, DataFrame, ndarray) or does not contain textual data.

    Examples
    --------
    # Assuming session_vars["parsed_logs"] contains a Series of log templates
    >>> vectorize_log_data(parsed_loglines="parsed_logs", algo_name="tfidf", save_as="log_vectors_tfidf")
    # session_vars["log_vectors_tfidf"] will store the resulting NumPy array.

    >>> vectorize_log_data(parsed_loglines="parsed_logs", algo_name="word2vec", vectorizer_params={"model_params": {"vector_size": 50}}, save_as="log_vectors_w2v")
    # session_vars["log_vectors_w2v"] will store vectors of size 50.

    See Also
    --------
    parse_log_data : Usually provides the `parsed_loglines` input.
    extract_log_features : Often takes these log vectors as input to combine
                           with other features.

    Notes
    -----
    - The `parsed_loglines` input is resolved using `_resolve`.
    - Input data is converted to a pandas Series of strings before processing.
    - The LogAI `LogVectorizer` is used for fitting and transforming the data.

    """
    if isinstance(parsed_loglines, (list, tuple)):
        series = pd.Series(parsed_loglines, dtype=str)
    elif isinstance(parsed_loglines, pd.Series):
        series = parsed_loglines.astype(str)
    elif isinstance(parsed_loglines, pd.DataFrame):
        series = parsed_loglines.iloc[:, 0].astype(str)
    elif isinstance(parsed_loglines, np.ndarray):
        series = pd.Series(parsed_loglines.flatten(), dtype=str)
    else:
        raise TypeError(
            "vectorize_log_data expects list/Series/DataFrame/numpy array; got "
            f"{type(parsed_loglines).__name__}"
        )

    cfg = VectorizerConfig()
    cfg.algo_name = algo_name  # type: ignore[attr-defined]
    for k, v in (vectorizer_params or {}).items():
        setattr(cfg, k, v)
    vec = LogVectorizer(cfg)
    vec.fit(series)
    vectors = np.asarray(vec.transform(series))

    return vectors


# ---------------------------------------------------------------------------
# Encode categorical attributes
# ---------------------------------------------------------------------------


def _encode_log_attributes_impl(
    attributes: Any,
    encoder_name: str = "label_encoder",
    encoder_params: dict | None = None,
) -> np.ndarray:
    """Encode categorical log attributes into a numerical NumPy array.

    Converts a DataFrame or Series of categorical attributes (e.g., IPs, status codes)
    into a numerical NumPy array using "label_encoder", "one_hot_encoder", etc.
    The resulting NumPy array is **mandatorily** stored in `session_vars` under the
    key provided in `save_as`. This named output is ready for use in ML models.

    Parameters
    ----------
    attributes : Any
        The categorical attributes to be encoded. This argument is resolved
        from `session_vars` if it's a string key. It must be, or be convertible to:
        - A pandas DataFrame where columns represent different categorical attributes.
        - A pandas Series representing a single categorical attribute.
        - A list of lists or list of dicts that can be converted to a DataFrame
          of categorical attributes.
        Example: `"log_attributes_df_var"` (a key in `session_vars`).
    encoder_name : str, default "label_encoder"
        The name of the encoding algorithm to use. Supported encoders depend on
        LogAI's `CategoricalEncoder`.
        Common options: "label_encoder", "one_hot_encoder".
        This argument is **not** resolved from `session_vars`.
    encoder_params : Dict | None, default None
        Optional dictionary of parameters specific to the chosen `encoder_name`.
        These are passed to LogAI's `CategoricalEncoderConfig`.
        This argument is **not** resolved from `session_vars`.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array containing the numerically encoded attributes.

    Side Effects
    ------------
    - Stores the NumPy array of numerically encoded attributes in `session_vars`
      under the key specified by `save_as` or an auto-generated key.
    - Raises `TypeError` if `attributes` is not of a supported type (DataFrame,
      Series, list) or cannot be converted to a suitable format for encoding.

    Examples
    --------
    # Assuming session_vars["attrs_df"] is a DataFrame with categorical columns
    >>> encode_log_attributes(attributes="attrs_df", encoder_name="one_hot_encoder", save_as="encoded_attrs_onehot")
    # session_vars["encoded_attrs_onehot"] will store the one-hot encoded NumPy array.

    # Using label encoding for a Series
    >>> session_vars["severity_series"] = pd.Series(['INFO', 'WARN', 'INFO', 'ERROR'])
    >>> encode_log_attributes(attributes="severity_series", encoder_name="label_encoder", save_as="encoded_severity")
    # session_vars["encoded_severity"] might store something like array([[0], [2], [0], [1]])

    See Also
    --------
    preprocess_log_data : Can provide the `attributes` DataFrame.
    extract_log_features : Often takes these encoded attributes as input to combine
                           with other features.

    Notes
    -----
    - The `attributes` input is resolved using `_resolve`.
    - Input data is converted to a pandas DataFrame internally if it's a Series
      or list.
    - LogAI's `CategoricalEncoder` is used for fitting and transforming the data.
      The `fit_transform` method returns a tuple `(encoded_df, fitted_encoder)`;
      this tool extracts and returns `encoded_df.to_numpy()`.
    - **Important**: This tool is for encoding *categorical* data. Do not pass
      already numerical data (e.g., feature vectors from `vectorize_log_data`
      or numerical columns like counts or measurements) to this tool. Doing so
      will likely lead to unintended behavior or errors.

    """
    if isinstance(attributes, pd.DataFrame):
        df = attributes
    elif isinstance(attributes, pd.Series):
        df = attributes.to_frame()
    elif isinstance(attributes, (list, tuple)):
        df = pd.DataFrame(attributes)
    else:
        raise TypeError(
            "encode_log_attributes expects DataFrame/Series/list; got "
            f"{type(attributes).__name__}"
        )

    enc_cfg = CategoricalEncoderConfig()
    enc_cfg.algo_name = encoder_name  # type: ignore[attr-defined]
    for k, v in (encoder_params or {}).items():
        setattr(enc_cfg, k, v)
    encoder = CategoricalEncoder(enc_cfg)
    encoded_df, _ = encoder.fit_transform(df)  # type: ignore[arg-type]
    encoded = encoded_df.to_numpy()

    return encoded


_SHELL.push(
    {
        "_vectorize_log_data_impl": _vectorize_log_data_impl,
        "_encode_log_attributes_impl": _encode_log_attributes_impl,
    }
)


@app.tool()
async def vectorize_log_data(
    parsed_loglines: Any,
    algo_name: str = "word2vec",
    vectorizer_params: dict | None = None,
    *,
    save_as: str,
):
    """Vectorize text and expose result via `save_as` in the shell.

    This wrapper delegates to `_vectorize_log_data_impl`.  The numerical matrix
    returned by that helper is bound to the *shell* variable name supplied via
    `save_as` so later tools can reference it.

    Parameters
    ----------
    parsed_loglines : Any
        The textual log data (e.g., templates, messages) to be vectorized.
        This argument is resolved from `session_vars` if it's a string key. It can be:
        - A pandas Series of template strings.
        - A pandas DataFrame where the first column contains template strings.
        - A list or tuple of template strings.
        - A NumPy array of template strings.
        Example: `"log_templates_var"` (a key in `session_vars`).
    algo_name : str, default "word2vec"
        The name of the vectorization algorithm to use. Supported algorithms
        depend on the LogAI library's `LogVectorizer`.
        Common options: "word2vec", "tfidf", "fasttext", "forecast_bert".
        This argument is **not** resolved from `session_vars`.
    vectorizer_params : Dict | None, default None
        Optional dictionary of parameters specific to the chosen `algo_name`.
        These are passed to LogAI's `VectorizerConfig`.
        Example for "word2vec": `{"model_params": {"vector_size": 100, "window": 5}}`.
        This argument is **not** resolved from `session_vars`.
    save_as : str
        Variable name that will receive the NumPy array inside the shell.

    Returns
    -------
    str
        The result of the shell command execution.

    Side Effects
    ------------
    - Stores the NumPy array of numerical feature vectors in `session_vars` under
      the key specified by `save_as` or an auto-generated key.
    - Raises `TypeError` if `parsed_loglines` is not of a supported type (list,
      Series, DataFrame, ndarray) or does not contain textual data.

    Examples
    --------
    # Assuming session_vars["parsed_logs"] contains a Series of log templates
    >>> vectorize_log_data(parsed_loglines="parsed_logs", algo_name="tfidf", save_as="log_vectors_tfidf")
    # session_vars["log_vectors_tfidf"] will store the resulting NumPy array.

    >>> vectorize_log_data(parsed_loglines="parsed_logs", algo_name="word2vec", vectorizer_params={"model_params": {"vector_size": 50}}, save_as="log_vectors_w2v")
    # session_vars["log_vectors_w2v"] will store vectors of size 50.

    See Also
    --------
    parse_log_data : Usually provides the `parsed_loglines` input.
    extract_log_features : Often takes these log vectors as input to combine
                           with other features.

    Notes
    -----
    - The `parsed_loglines` input is resolved using `_resolve`.
    - Input data is converted to a pandas Series of strings before processing.
    - The LogAI `LogVectorizer` is used for fitting and transforming the data.

    """
    code = f"{save_as} = _vectorize_log_data_impl({repr(parsed_loglines)}, {repr(algo_name)}, {repr(vectorizer_params)})\n"
    return await run_code_in_shell(code)


vectorize_log_data.__doc__ = _vectorize_log_data_impl.__doc__


@app.tool()
async def encode_log_attributes(
    attributes: Any,
    encoder_name: str = "label_encoder",
    encoder_params: dict | None = None,
    *,
    save_as: str,
):
    """Encode categorical attributes and bind the array via `save_as`.

    Delegates to `_encode_log_attributes_impl` and assigns the returned NumPy
    array to the provided `save_as` variable name inside the IPython shell.

    Parameters
    ----------
    attributes : Any
        The categorical attributes to be encoded. This argument is resolved
        from `session_vars` if it's a string key. It must be, or be convertible to:
        - A pandas DataFrame where columns represent different categorical attributes.
        - A pandas Series representing a single categorical attribute.
        - A list of lists or list of dicts that can be converted to a DataFrame
          of categorical attributes.
        Example: `"log_attributes_df_var"` (a key in `session_vars`).
    encoder_name : str, default "label_encoder"
        The name of the encoding algorithm to use. Supported encoders depend on
        LogAI's `CategoricalEncoder`.
        Common options: "label_encoder", "one_hot_encoder".
        This argument is **not** resolved from `session_vars`.
    encoder_params : Dict | None, default None
        Optional dictionary of parameters specific to the chosen `encoder_name`.
        These are passed to LogAI's `CategoricalEncoderConfig`.
        This argument is **not** resolved from `session_vars`.
    save_as : str
        Variable name that will hold the encoded attribute matrix.

    Returns
    -------
    str
        The result of the shell command execution.

    Side Effects
    ------------
    - Stores the NumPy array of numerically encoded attributes in `session_vars`
      under the key specified by `save_as` or an auto-generated key.
    - Raises `TypeError` if `attributes` is not of a supported type (DataFrame,
      Series, list) or cannot be converted to a suitable format for encoding.

    Examples
    --------
    # Assuming session_vars["attrs_df"] is a DataFrame with categorical columns
    >>> encode_log_attributes(attributes="attrs_df", encoder_name="one_hot_encoder", save_as="encoded_attrs_onehot")
    # session_vars["encoded_attrs_onehot"] will store the one-hot encoded NumPy array.

    # Using label encoding for a Series
    >>> session_vars["severity_series"] = pd.Series(['INFO', 'WARN', 'INFO', 'ERROR'])
    >>> encode_log_attributes(attributes="severity_series", encoder_name="label_encoder", save_as="encoded_severity")
    # session_vars["encoded_severity"] might store something like array([[0], [2], [0], [1]])

    See Also
    --------
    preprocess_log_data : Can provide the `attributes` DataFrame.
    extract_log_features : Often takes these encoded attributes as input to combine
                           with other features.

    Notes
    -----
    - The `attributes` input is resolved using `_resolve`.
    - Input data is converted to a pandas DataFrame internally if it's a Series
      or list.
    - LogAI's `CategoricalEncoder` is used for fitting and transforming the data.
      The `fit_transform` method returns a tuple `(encoded_df, fitted_encoder)`;
      this tool extracts and returns `encoded_df.to_numpy()`.
    - **Important**: This tool is for encoding *categorical* data. Do not pass
      already numerical data (e.g., feature vectors from `vectorize_log_data`
      or numerical columns like counts or measurements) to this tool. Doing so
      will likely lead to unintended behavior or errors.

    """
    code = f"{save_as} = _encode_log_attributes_impl({repr(attributes)}, {repr(encoder_name)}, {repr(encoder_params)})\n"
    return await run_code_in_shell(code)


encode_log_attributes.__doc__ = _encode_log_attributes_impl.__doc__
