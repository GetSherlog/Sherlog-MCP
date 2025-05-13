#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "logai",
#   "mcp",
#   "fastmcp",
#   "pandas",
#   "numpy",
#   "scikit-learn",
#   "nltk",
#   "Cython",
#   "requests",
#   "scipy<2.0",
#   "gensim"
# ]
# requires-python = ">=3.10"
# ///

from typing import List, Any, Dict, Union, Literal
import logging
import functools

import pandas as pd
import nltk
import nltk.downloader

from mcp.server.fastmcp import FastMCP

from logai.dataloader.data_loader import DataLoaderConfig, FileDataLoader

from logai.preprocess.preprocessor import PreprocessorConfig, Preprocessor
from logai.utils import constants


from logai.information_extraction.log_parser import LogParser, LogParserConfig
from logai.algorithms.parsing_algo.drain import DrainParams
from logai.information_extraction.log_vectorizer import VectorizerConfig, LogVectorizer
from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig, CategoricalEncoder
from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor
from logai.algorithms.clustering_algo.kmeans import KMeansParams
from logai.analysis.clustering import ClusteringConfig, Clustering


# Initialize FastMCP server
app = FastMCP(name="LogAIMCP", dependencies=[
    "logai",
    "pandas",
    "numpy",
    "scipy<2.0",
    "gensim"
])

# ----------------------------------------------------
# Download NLTK resources if missing
# ----------------------------------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
# Specifically for punkt_tab, though 'punkt' usually includes it.
# Explicitly check and download 'punkt_tab' if still needed or if 'punkt' is too broad.
# However, NLTK's standard 'punkt' download should cover common tokenization needs.
# If 'punkt_tab' is a distinct package and consistently missing,
# an explicit download like nltk.download('punkt_tab', quiet=True) would be added here.
# For now, relying on the main 'punkt' package.


# ----------------------------------------------------
# Simple in-memory scratch-pad that lets tools share
# intermediate results across a single FastMCP session
# ----------------------------------------------------
session_vars: Dict[str, Any] = {}

def _resolve(arg: Any) -> Any:
    """Return the stored object if *arg* is a key in *session_vars*.

    This allows tools to accept either the concrete value or a string
    that refers to the name under which that value was stored by a
    previous tool (via the *save_as* option).
    """
    return session_vars.get(arg, arg) if isinstance(arg, str) else arg


# -------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------

logger = logging.getLogger("LogAIMCP")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def _short_repr(obj: Any, max_len: int = 200) -> str:
    """Return a safe, truncated ``repr`` for logging purposes."""
    try:
        r = repr(obj)
        if len(r) > max_len:
            r = r[:max_len] + "..."
        return r
    except Exception:
        return f"<{type(obj).__name__}>"


def log_tool(func):
    """Decorator that logs entry and exit of FastMCP tools."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info("→ %s args=%s kwargs=%s", func.__name__, _short_repr(args), _short_repr(kwargs))
        try:
            result = func(*args, **kwargs)
            logger.info("← %s result=%s", func.__name__, _short_repr(result))
            return result
        except Exception as exc:
            logger.exception("! %s raised %s", func.__name__, exc)
            raise

    return wrapper


# Load log data

@app.tool()
@log_tool
def get_log_file_columns(file_path: str, *, save_as: str | None = None) -> List[str]:
    """Return the column names in a CSV-style log file.

    Parameters
    ----------
    file_path : str | SessionVar
        Absolute/relative path to the log file **or** the *name* of a value
        that was previously stored with ``save_as``.  The helper
        :pyfunc:`_resolve` automatically swaps the name for the stored
        object when the argument is a ``str`` that matches a key in
        ``session_vars``.
    save_as : str, optional
        If provided, the resulting ``List[str]`` is placed in
        ``session_vars[save_as]`` so that later tool calls can refer to
        it just by the *name*.

    Returns
    -------
    list[str]
        The list of column names.

    Examples
    --------
    >>> get_log_file_columns("logs.csv", save_as="cols")
    ['timestamp', 'body', 'level', ...]
    >>> # use the stored value in a Python expression
    >>> py("v['cols'][:3]")
    ['timestamp', 'body', 'level']

    See Also
    --------
    list_session_vars, py, suggest_dimension_mapping
    """
    file_path = _resolve(file_path)
    cols = pd.read_csv(file_path).columns.tolist()
    if save_as:
        session_vars[save_as] = cols
    return cols


@app.tool()
@log_tool
def load_file_log_data(file_path: Any, dimensions: dict, 
    log_type: str = "csv",
    infer_datetime: bool = False,
    datetime_format: str = "%Y-%m-%dT%H:%M:%SZ",
    *,
    save_as: str | None = None) -> Any:
    """Load structured log data from a local file.

    This is the usual *entry-point* when starting an analysis session.
    The loaded dataset can be captured under a variable name with
    ``save_as`` and passed to later tools (e.g. ``preprocess_log_data``)
    simply by using that variable name.

    Parameters
    ----------
    file_path : str | SessionVar
        Path or previously-saved variable that points to the file.
    dimensions : dict
        Mapping that tells *LogAI* which columns correspond to
        ``timestamp``, ``body`` etc.  Example::

            {"timestamp": ["ts"], "body": ["message"]}

    log_type : {"csv", "json", ...}, default "csv"
        Loader backend to use.
    infer_datetime : bool, default False
        Whether to have LogAI parse the timestamp column
        automatically.
    datetime_format : str, default "%Y-%m-%dT%H:%M:%SZ"
        ``strptime`` format to fall back to when ``infer_datetime``
        is False.
    save_as : str, optional
        Persist the resulting *LogRecordObject* as
        ``session_vars[save_as]``.

    Returns
    -------
    Any
        LogAI record object – typically a ``Dataset`` like wrapper.

    Example
    -------
    >>> load_file_log_data("logs.csv", dim_map, save_as="raw")
    >>> preprocess_log_data("raw", custom_replace, save_as="clean")

    See Also
    --------
    list_session_vars, py, suggest_dimension_mapping

    Scenarios
    ---------
    **1. Happy path**::

        cols = get_log_file_columns("weblog.csv")
        # -> ['timestamp', 'URL', 'status_code', ...]

        dim = {"timestamp": ["timestamp"], "body": ["URL"]}
        logs = load_file_log_data("weblog.csv", dim, save_as="raw")

    **2. Column-name typo (raises)**::

        dim = {"timestamp": ["IPTime"], "body": ["URL"]}
        # ValueError: Dimension mapping refers to columns that do not exist: ['IPTime'] | Available columns: [...]
        load_file_log_data("weblog.csv", dim)
        # → fix the mapping using actual column names.

    Notes
    -----
    The object returned is a :class:`logai.dataloader.data_model.LogRecordObject`.
    It exposes dataframes as *attributes* (e.g. ``raw.body``) – it does **not**
    behave like a dictionary, so avoid calls such as ``raw.keys()``.  Use
    :pyfunc:`describe_log_record` to introspect its structure.
    """
    file_path = _resolve(file_path)

    # --- Validate that provided dimension columns exist ----------------
    header_cols = pd.read_csv(file_path, nrows=0).columns
    missing_cols = {
        dim_col
        for dim_list in dimensions.values() for dim_col in dim_list
        if dim_col not in header_cols
    }
    if missing_cols:
        raise ValueError(
            "Dimension mapping refers to columns that do not exist: "
            f"{sorted(missing_cols)} | Available columns: {list(header_cols)}"
        )

    data_loader = FileDataLoader(DataLoaderConfig(
        filepath=file_path,
        log_type=log_type,
        infer_datetime=infer_datetime,
        datetime_format=datetime_format,
        dimensions=dimensions
    ))
    result = data_loader.load_data()
    if save_as:
        session_vars[save_as] = result
    return result


# Preprocess log data

@app.tool()
@log_tool
def preprocess_log_data(log_record: Any, custom_replace_list: List[List[str]], *, save_as: str | None = None,
    save_clean_as: str | None = None,
    save_patterns_as: str | None = None,
    save_attributes_as: str | None = None):
    """Clean raw log lines and optionally apply custom regex replacements.

    Parameters
    ----------
    log_record : LogAIRecord | SessionVar
        Object returned by :pyfunc:`load_file_log_data` **or** the name
        under which it was stored earlier.
    custom_replace_list : list[list[str]]
        Each inner list contains ``[pattern, replacement]``.  These
        regexes are fed into :class:`logai.preprocess.Preprocessor`.
    save_as : str, optional
        Store the *tuple* ``(clean_logs, custom_patterns, attributes)``
        in ``session_vars[save_as]``.
        **Note:** For easier access to individual components, it is highly
        recommended to use ``save_clean_as``, ``save_patterns_as``, and
        ``save_attributes_as`` instead.
    save_clean_as, save_patterns_as, save_attributes_as : str, optional
        If provided, store the three individual outputs (cleaned log lines,
        custom patterns, and attributes DataFrame) under their respective
        names in session_vars. This is the **recommended** way to save outputs.

    Returns
    -------
    tuple
        A tuple containing ``(clean_logs, custom_patterns, attributes)``.
        - ``clean_logs``: The cleaned log lines.
        - ``custom_patterns``: The custom patterns applied.
        - ``attributes``: A pandas DataFrame of attributes from the log record.

    Example
    -------
    >>> # Recommended: Save individual components
    >>> preprocess_log_data("raw", [[r"\\d+", "<NUM>"]],
    ...                     save_clean_as="cleaned_lines",
    ...                     save_attributes_as="log_attrs")
    >>> # list_session_vars() would show 'cleaned_lines', 'log_attrs'

    >>> # Legacy: Save as a tuple
    >>> preprocess_log_data("raw", [[r"\\d+", "<NUM>"]], save_as="preprocessed_tuple")
    >>> # To access attributes if saved as a tuple:
    >>> # py("v['preprocessed_tuple'][2]", save_as="attributes_from_tuple")
    >>> # Then use "attributes_from_tuple" in subsequent tools.


    See Also
    --------
    list_session_vars, py, load_file_log_data
    """
    log_record = _resolve(log_record)
    loglines = log_record.body[constants.LOGLINE_NAME]
    attributes = log_record.attributes

    preprocessor_config = PreprocessorConfig(
        custom_replace_list=custom_replace_list
    )

    preprocessor = Preprocessor(preprocessor_config)

    clean_logs, custom_patterns = preprocessor.clean_log(
        loglines
    )

    # ------------------------------------------------------------------
    # Persist results according to caller preference
    # ------------------------------------------------------------------

    if save_clean_as:
        session_vars[save_clean_as] = clean_logs
    if save_patterns_as:
        session_vars[save_patterns_as] = custom_patterns
    if save_attributes_as:
        session_vars[save_attributes_as] = attributes

    # Legacy combined save
    result = (clean_logs, custom_patterns, attributes)
    if save_as:
        session_vars[save_as] = result

    return result



@app.tool()
@log_tool
def parse_log_data(clean_logs: Any, parsing_algorithm: str = "drain", *, save_as: str | None = None) -> Any:
    """Transform cleaned log lines into structured templates.

    Parameters
    ----------
    clean_logs : Iterable[str] | SessionVar
        Clean log lines returned by :pyfunc:`preprocess_log_data`, or
        the *name* of a stored value.
    parsing_algorithm : {"drain", ...}, default "drain"
        Which parsing backend to employ.
        
        *When using the default ``drain`` backend it is **normal** to
        occasionally see warnings like ``NoneType has no attribute
        'log_template_tokens'`` for very noisy datasets.*  These stem
        from the underlying Drain implementation returning ``None``
        for log lines that do not match any previously-learned
        template.  The wrapper now **catches** such conditions and
        falls back to a "no-op" parse (the raw line is returned as
        the template) so that downstream steps can continue instead
        of failing hard.
    save_as : str, optional
        Persist the parsed log lines for later steps.

    Returns
    -------
    Any
        Parsed log lines – depends on the chosen algorithm.

    Example
    -------
    >>> parse_log_data("clean", save_as="parsed")

    See Also
    --------
    list_session_vars, py, suggest_dimension_mapping
    """
    clean_logs = _resolve(clean_logs)

    # Accept the tuple produced by `preprocess_log_data` directly
    if isinstance(clean_logs, tuple) and len(clean_logs) >= 1:
        clean_logs = clean_logs[0]  # first element is the cleaned log lines

    # Convert pandas Series / DataFrame to list[str]
    try:
        import pandas as _pd
        if isinstance(clean_logs, _pd.Series):
            clean_logs = clean_logs.astype(str).tolist()
        elif isinstance(clean_logs, _pd.DataFrame):
            # assume first column contains log text
            clean_logs = clean_logs.iloc[:, 0].astype(str).tolist()
    except Exception:
        pass  # If pandas isn't available or other issue, proceed as-is

    parsing_algo_params = DrainParams(
    sim_th=0.5, depth=5
    )

    log_parser_config = LogParserConfig(
        parsing_algorithm=parsing_algorithm,
        parsing_algo_params=parsing_algo_params
    )

    parser = LogParser(log_parser_config)
    try:
        parsed_result = parser.parse(clean_logs)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover – best-effort resilience
        # Drain may raise AttributeError ("NoneType has no attribute ...")
        # when a log line cannot be clustered.  Instead of surfacing a
        # confusing stack-trace to the LLM we log & fall back to an
        # identity transformation so the pipeline can keep going.
        logger.warning(
            "Falling back to raw lines because parsing failed: %s: %s",
            type(exc).__name__, exc,
        )
        try:
            import pandas as _pd
            parsed_result = _pd.Series(clean_logs)  # type: ignore[arg-type]
        except Exception:
            parsed_result = list(clean_logs)

    # LogAI's ``parse`` may return different structures depending on the
    # underlying algorithm / version.  We *prefer* the ``parsed_logline``
    # column (DataFrame) when available, but fall back gracefully to the
    # raw ``parsed_result`` otherwise so we don't crash with mysterious
    # ``NoneType`` errors.
    if parsed_result is None:
        # Surface a clearer diagnostics message instead of the cryptic
        # "'NoneType'" that would result from subscripting ``None``.
        raise ValueError("Log parsing returned None – check that the input ``clean_logs`` is a non-empty iterable of strings.")

    try:
        parsed_loglines = parsed_result["parsed_logline"]  # type: ignore[index]
    except (TypeError, KeyError):
        # Either the result is not subscriptable (e.g., list) or the
        # expected column is missing.  In that case, just propagate the
        # entire object so that downstream steps can still inspect it.
        parsed_loglines = parsed_result

    if save_as:
        session_vars[save_as] = parsed_loglines
    return parsed_loglines


# -------------------------------------------------------------------
# Vectorize parsed log templates
# -------------------------------------------------------------------


@app.tool()
@log_tool
def vectorize_log_data(
    parsed_loglines: Any,
    algo_name: str = "word2vec",
    vectorizer_params: dict | None = None,
    *,
    save_as: str | None = None,
) -> Any:
    """Convert parsed log templates into numerical feature vectors.

    Parameters
    ----------
    parsed_loglines : Iterable[str] | pandas.Series | pandas.DataFrame | SessionVar
        Output from :pyfunc:`parse_log_data` or the name under which it was stored.
    algo_name : {"word2vec", "tfidf", ...}, default "word2vec"
        Vectorization algorithm to use.  Passed through to
        :class:`logai.information_extraction.log_vectorizer.VectorizerConfig`.
    vectorizer_params : dict, optional
        Additional keyword arguments forwarded to ``VectorizerConfig``.
    save_as : str, optional
        Persist the resulting vectors (``numpy.ndarray``) under this name.

    Returns
    -------
    Any
        Typically a ``numpy.ndarray`` of shape (n_samples, n_features).

    Example
    -------
    >>> vectorize_log_data("parsed_logs", save_as="vectors")
    >>> py("v['vectors'].shape")
    (16007, 100)
    """

    import numpy as _np
    import pandas as _pd

    parsed_loglines = _resolve(parsed_loglines)

    # --- Input Validation for LLM ---
    if not isinstance(parsed_loglines, (_pd.Series, _pd.DataFrame, list, tuple, _np.ndarray)):
        err_msg = (
            f"The 'parsed_loglines' argument received an unexpected type: {type(parsed_loglines).__name__}. "
            "It should be a pandas Series, DataFrame, list, tuple, or numpy array of log lines. "
            "This often occurs if the session variable name is incorrect or if the preceding step "
            "(e.g., 'parse_log_data') did not produce the expected output. "
            "Please verify the variable name and the output of the 'parse_log_data' step."
        )
        logger.error(err_msg)
        raise TypeError(err_msg)
    if isinstance(parsed_loglines, str): # Should be caught by the above, but as an explicit check
        err_msg = (
            f"The 'parsed_loglines' argument was resolved to a string: '{parsed_loglines[:100]}{'...' if len(parsed_loglines) > 100 else ''}'. "
            "This is not a valid input. It should be a collection of parsed log lines (e.g., list, pandas Series). "
            "Ensure the session variable name provided for 'parsed_loglines' correctly refers to the output of 'parse_log_data'."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)


    # Convert various input types to a pandas Series of strings,
    # as expected by LogAI's LogVectorizer.
    if isinstance(parsed_loglines, (list, tuple)):
        parsed_loglines_series = _pd.Series(parsed_loglines, dtype=str)
    elif isinstance(parsed_loglines, _pd.DataFrame):
        if parsed_loglines.empty:
            parsed_loglines_series = _pd.Series([], dtype=str)
        else:
            # Assuming the first column contains the log text if a DataFrame is passed.
            parsed_loglines_series = parsed_loglines.iloc[:, 0].astype(str)
    elif isinstance(parsed_loglines, _np.ndarray):
        # Flatten and convert to string Series if it's a numpy array
        parsed_loglines_series = _pd.Series(parsed_loglines.flatten(), dtype=str)
    elif isinstance(parsed_loglines, _pd.Series):
        parsed_loglines_series = parsed_loglines.astype(str) # Ensure it's a Series of strings
    else:
        # This case should ideally be caught by the initial type checks,
        # but as a fallback, attempt direct conversion if it's some other Series-like.
        try:
            parsed_loglines_series = _pd.Series(parsed_loglines, dtype=str)
        except Exception as e:
            final_err_msg = (
                f"Could not convert 'parsed_loglines' of type {type(parsed_loglines).__name__} to a pandas Series of strings. Error: {e}. "
                "Please ensure 'parsed_loglines' is a list of strings, pandas Series/DataFrame, or numpy array."
            )
            logger.error(final_err_msg)
            raise TypeError(final_err_msg) from e


    vectorizer_params = vectorizer_params or {}

    vec_cfg = VectorizerConfig(algo_name=algo_name, **vectorizer_params)
    vectorizer = LogVectorizer(vec_cfg)

    try:
        vectorizer.fit(parsed_loglines_series)
        log_vectors = vectorizer.transform(parsed_loglines_series)
    except LookupError as e:
        if "punkt_tab" in str(e):
            # Make the error message more actionable for an LLM guiding a user
            error_message = (
                "NLTK resource 'punkt_tab' not found, which is required for tokenization by the 'tfidf' or 'word2vec' vectorizer. "
                "To resolve this, the user needs to download this resource. "
                "Please instruct the user to run the following Python code in their environment: "
                "import nltk; nltk.download('punkt_tab')"
            )
            logger.error(error_message)
            raise LookupError(error_message) from e
        if "punkt" in str(e): # General punkt error, also guide for LLM
            error_message = (
                "NLTK resource 'punkt' not found, which is required for tokenization. "
                "To resolve this, the user needs to download this resource. "
                "Please instruct the user to run the following Python code in their environment: "
                "import nltk; nltk.download('punkt')"
            )
            logger.error(error_message)
            raise LookupError(error_message) from e
        raise  # Re-raise other LookupErrors

    # Ensure result is numpy array for consistent downstream usage
    log_vectors = _np.asarray(log_vectors)

    if save_as:
        session_vars[save_as] = log_vectors
    return log_vectors


# -------------------------------------------------------------------
# Categorical encoding for log attributes
# -------------------------------------------------------------------


@app.tool()
@log_tool
def encode_log_attributes(
    attributes: Any,
    encoder_name: str = "label_encoder",
    encoder_params: dict | None = None,
    *,
    save_as: str | None = None,
) -> Any:
    """Encode categorical log *attributes* into numerical representations.

    Parameters
    ----------
    attributes : pandas.DataFrame | pandas.Series | Iterable | SessionVar
        The attributes field returned by :pyfunc:`load_file_log_data` or
        :pyfunc:`preprocess_log_data`, or a *name* referring to a stored
        value in *session_vars*.
    encoder_name : str, default "label_encoder"
        Name assigned to the underlying :class:`CategoricalEncoderConfig`.
    encoder_params : dict, optional
        Additional keyword arguments forwarded to
        ``CategoricalEncoderConfig`` for fine-tuning the encoder.
    save_as : str, optional
        Persist the encoded attributes under this variable name.

    Returns
    -------
    Any
        The encoded attribute matrix (typically ``numpy.ndarray``).
    """

    import numpy as _np
    import pandas as _pd

    attributes = _resolve(attributes)

    # --- Input Validation for LLM ---
    if isinstance(attributes, str):
        # Check if it's a string that might be an unresolved expression
        if "[" in attributes and "]" in attributes and ("v[" in attributes or "session_vars[" in attributes):
            error_message = (
                f"The 'attributes' argument was resolved to a string expression: '{attributes}'. "
                "This tool expects the actual data (e.g., a pandas DataFrame or Series), not a string expression. "
                "To resolve this: "
                "1. Use the 'py' tool to evaluate this expression and save its result to a new session variable. "
                "   Example: py(expr=\"{attributes}\", save_as=\"my_attributes_data\") "
                "   Then, pass \"my_attributes_data\" as the 'attributes' argument. "
                "2. If the attributes data comes from 'preprocess_log_data', it is highly recommended to use "
                "   its 'save_attributes_as' parameter (e.g., save_attributes_as=\"log_attributes\") to directly save "
                "   the attributes DataFrame. Then use \"log_attributes\" here."
            )
        else:
            error_message = (
                f"The 'attributes' argument was resolved to a string: '{attributes}'. "
                "This tool expects a pandas DataFrame or Series, or a list/tuple that can be converted to one. "
                "Please ensure the session variable name you provided for 'attributes' is correct and "
                "that it holds the actual attributes data, not just its name as a string. "
                "If the data comes from 'preprocess_log_data', check if you intended to use its output; "
                "consider using its 'save_attributes_as' parameter for direct access."
            )
        logger.error(error_message)
        raise ValueError(error_message)

    if not isinstance(attributes, (_pd.DataFrame, _pd.Series, list, tuple, _np.ndarray)):
        error_message = (
            f"The 'attributes' argument received an unexpected type: {type(attributes).__name__}. "
            "It should be a pandas DataFrame, Series, list, tuple, or numpy array. "
            "This often occurs if the session variable name is incorrect or if the preceding "
            "step (e.g., 'preprocess_log_data' using 'save_attributes_as', or 'py' tool to extract from a tuple) "
            "did not produce the expected output or was not used correctly. Please verify the input."
        )
        logger.error(error_message)
        raise TypeError(error_message)
    # --- End Input Validation ---


    # Convert plain lists / tuples into a DataFrame for the encoder
    if isinstance(attributes, (list, tuple)):
        # Attempt to convert to DataFrame, but catch if it's not suitable (e.g. list of strings)
        try:
            attributes_df = _pd.DataFrame(attributes)
            if attributes_df.empty and len(attributes) > 0: # e.g. list of scalars
                 attributes_df = _pd.DataFrame({"column_from_list": attributes})

        except ValueError as e:
            error_message = (
                f"Failed to convert the 'attributes' (currently a {type(attributes).__name__}) to a pandas DataFrame. Error: {e}. "
                "Ensure the list/tuple contains data suitable for DataFrame conversion (e.g., a list of lists/dicts, not just a flat list of strings unless it's for a single column). "
                "If it's a flat list intended as a single column, the tool will attempt to wrap it. "
                "Check the structure of your attributes data."
            )
            logger.error(error_message)
            # If it's a simple list of scalars, it might be intended as a single column.
            # The encoder itself might handle a Series better or we make it a single-column DF.
            try:
                attributes_df = _pd.DataFrame({"auto_column": attributes})
            except Exception: # Fallback if even that fails
                 raise ValueError(error_message) from e

    elif isinstance(attributes, _pd.Series):
        attributes_df = attributes.to_frame() # Convert Series to DataFrame
    else:
        attributes_df = attributes  # expected DataFrame / numpy array which will be converted by encoder

    encoder_params = encoder_params or {}

    enc_cfg = CategoricalEncoderConfig(name=encoder_name, **encoder_params)
    encoder = CategoricalEncoder(enc_cfg)

    encoded = encoder.fit_transform(attributes_df)  # type: ignore[arg-type]

    # Ensure numpy array for consistency
    encoded = _np.asarray(encoded)

    if save_as:
        session_vars[save_as] = encoded
    return encoded


# -------------------------------------------------------------------
# Light-weight REPL tool – evaluate arbitrary Python expressions
# -------------------------------------------------------------------

@app.tool()
@log_tool
def py(expr: str, *, save_as: str | None = None):
    """Run arbitrary Python code against the current session.

    The *expr* string is first tried with ``eval`` (expression mode).  If
    that raises ``SyntaxError`` we fall back to ``exec`` (statement
    mode), which lets you write multi-line blocks, ``import``
    statements, ``for`` loops, etc.

    A variable ``v`` is injected into the execution context and points
    to the session variable dictionary.

    Parameters
    ----------
    expr : str
        Valid Python code.  For single expressions you can omit the
        trailing ``return``; for multi-statement code the result of the
        last expression assigned to ``_`` (underscore) will be returned
        if set.
    save_as : str, optional
        Assign the result to ``session_vars[save_as]`` if provided.

    Examples
    --------
    >>> py("v['parsed'][:10]")               # preview first 10 rows
    >>> py("{ 'n_rows': len(v['parsed']) }", save_as="meta")

    See Also
    --------
    list_session_vars, suggest_dimension_mapping
    """
    v = session_vars  # shorthand for the expression context
    try:
        # ----------------------------------------------------------
        # First attempt: treat *expr* as a single Python expression
        # ----------------------------------------------------------
        code_obj = compile(expr, "<py-eval>", "eval")
        result = eval(code_obj, {}, {"v": v})
    except SyntaxError:
        # ----------------------------------------------------------
        # Multi-statement block → use exec() semantics instead
        # ----------------------------------------------------------
        local_env = {"v": v}
        try:
            code_obj = compile(expr, "<py-exec>", "exec")
            exec(code_obj, {}, local_env)
            result = local_env.get("_", None)
        except Exception as exc:
            # Capture runtime errors (e.g. NameError, TypeError) so that the
            # calling LLM receives a *value* instead of a stack-trace that
            # aborts the entire conversation.
            logger.warning("py tool execution failed: %s: %s", type(exc).__name__, exc)
            result = f"[PY-ERROR] {type(exc).__name__}: {exc}"
    except Exception as exc:
        # Catch evaluation-time errors from the initial eval() attempt.
        logger.warning("py tool execution failed: %s: %s", type(exc).__name__, exc)
        result = f"[PY-ERROR] {type(exc).__name__}: {exc}"

    if save_as:
        session_vars[save_as] = result
    return result


# -------------------------------------------------------------------
# Utility: list current session variables
# -------------------------------------------------------------------

@app.tool()
@log_tool
def list_session_vars(show_values: bool = False):
    """Inspect what objects are currently stored in *session_vars*.

    Parameters
    ----------
    show_values : bool, default False
        If *False* (default) only the names of stored variables are
        returned.  If *True* the full ``dict`` mapping names to values
        is returned.

    Returns
    -------
    list[str] | dict[str, Any]
        Either the list of variable names or the full mapping.

    Examples
    --------
    >>> list_session_vars()
    ['raw', 'clean']
    >>> list_session_vars(True)
    {'raw': <LogRecord>, 'clean': (...)}

    See Also
    --------
    py, suggest_dimension_mapping
    """
    return session_vars if show_values else list(session_vars.keys())


# -------------------------------------------------------------------
# Heuristic helper to build an initial dimensions mapping
# -------------------------------------------------------------------

def _default_dimension_mapping(cols: List[str]) -> dict:
    # Normalize column names for matching purposes
    def _norm(name: str) -> str:
        """Return a lowercase identifier with punctuation removed for fuzzy matching."""
        return (
            name.lower()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

    ts_candidates = [c for c in cols if _norm(c) in {"timestamp", "time", "date", "datetime"}]

    body_candidates_exact = {"body", "message", "url", "log", "text"}
    body_candidates = [c for c in cols if _norm(c) in body_candidates_exact]

    # ------------------------------------------------------------------
    # Heuristic label detection
    # ------------------------------------------------------------------
    # "Labels" in LogAI carry ground-truth class information.  Common
    # names we see in practice include: label, labels, status,
    # response_status, severity, level, class, outcome, anomaly.
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

    mapping = {
        "timestamp": ts_candidates[:1],
        "body": body_candidates[:1] if body_candidates else [],
        "span_id": [],
        "labels": label_candidates[:1],
    }

    return mapping


@app.tool()
@log_tool
def suggest_dimension_mapping(file_path: str, *, save_as: str | None = None) -> dict:
    """Generate a starter ``dimensions`` mapping for *file_path*.

    Uses simple heuristics on column names (e.g., columns containing
    'time' go to ``timestamp``) so the agent gets a valid structure that
    can be tweaked rather than invented from scratch.

    Parameters
    ----------
    file_path : str
        Path to the CSV (or other structured log) file.
    save_as : str, optional
        If given, store the resulting mapping under this name.

    Returns
    -------
    dict
        Skeleton mapping compliant with LogAI.

    Example
    -------
    >>> dim = suggest_dimension_mapping("weblog.csv", save_as="dim")
    >>> load_file_log_data("weblog.csv", dim)
    """
    cols = get_log_file_columns(file_path)
    mapping = _default_dimension_mapping(cols)
    if save_as:
        session_vars[save_as] = mapping
    return mapping


# -------------------------------------------------------------------
# Describe LogRecordObject helper
# -------------------------------------------------------------------

@app.tool()
@log_tool
def describe_log_record(record: Any, sample_rows: int = 5, *, save_as: str | None = None) -> dict:
    """Return a summary of a :class:`LogRecordObject`.

    Parameters
    ----------
    record : LogRecordObject | SessionVar
        The log record object to inspect, or the name under which it was stored.
    sample_rows : int, default 5
        How many rows to include from each dataframe field for preview.
    save_as : str, optional
        Store the resulting summary dict under this variable name.

    Returns
    -------
    dict
        Mapping each top-level attribute to information such as number of
        rows, column names, and an optional head preview.
    """
    record = _resolve(record)

    summary: dict = {}
    import pandas as _pd

    for field_name in (
        "timestamp",
        "attributes",
        "resource",
        "trace_id",
        "span_id",
        "severity_text",
        "severity_number",
        "body",
        "labels",
    ):
        df = getattr(record, field_name, None)
        if isinstance(df, _pd.DataFrame):
            summary[field_name] = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "head": df.head(sample_rows).to_dict(orient="list"),
            }
    if save_as:
        session_vars[save_as] = summary
    return summary


# -------------------------------------------------------------------
# Inspect Session Variable tool
# -------------------------------------------------------------------

@app.tool()
@log_tool
def inspect_session_variable(
    variable_name: str,
    preview_elements: int = 5,
    max_preview_str_len: int = 500
) -> Dict[str, Any]:
    """Inspect a session variable to understand its type, structure, and content.

    Provides detailed information like type, length, shape (for arrays/DataFrames),
    data types (for pandas objects), and a preview of the content.

    Parameters
    ----------
    variable_name : str
        The name of the variable stored in session_vars to inspect.
    preview_elements : int, default 5
        Number of elements/rows to include in the preview for sequences or DataFrames.
    max_preview_str_len : int, default 500
        Maximum length for the string representation of the preview to prevent
        excessively long outputs.

    Returns
    -------
    dict
        A dictionary containing inspection details:
        - 'name': Name of the variable.
        - 'type': String representation of the variable's type.
        - 'exists': Boolean, true if variable_name is in session_vars.
        - 'length': Length of the object (if applicable, e.g., list, dict, pd.Series/DataFrame rows).
        - 'shape': Shape of the object (if applicable, e.g., pd.DataFrame, np.ndarray).
        - 'dtypes': Data types (for pd.DataFrame columns, pd.Series, or np.ndarray).
        - 'preview': A string preview of the variable's content.
        - 'error': An error message if inspection failed (e.g., variable not found).

    Example
    -------
    >>> # Assuming 'raw_logs' (a DataFrame) and 'my_list' are in session_vars
    >>> inspect_session_variable("raw_logs")
    {
        "name": "raw_logs",
        "type": "<class 'pandas.core.frame.DataFrame'>",
        "exists": True,
        "length": 1000,
        "shape": [1000, 5],
        "dtypes": {"timestamp": "datetime64[ns]", "message": "object"},
        "preview": "   timestamp      message\\n0  2023-01-01  Log entry 1\\n1  2023-01-01  Log entry 2..."
    }
    >>> inspect_session_variable("my_list")
    {
        "name": "my_list",
        "type": "<class 'list'>",
        "exists": True,
        "length": 10,
        "preview": "[1, 2, 3, 4, 5, '... (5 more items)']"
    }
    >>> inspect_session_variable("non_existent_var")
    {
        "name": "non_existent_var",
        "type": "N/A",
        "exists": False,
        "error": "Variable 'non_existent_var' not found in session variables."
    }
    """
    import pandas as _pd
    import numpy as _np

    details: Dict[str, Any] = {
        "name": variable_name,
        "type": "N/A",
        "exists": False,
        "length": None,
        "shape": None,
        "dtypes": None,
        "preview": "N/A",
        "error": None
    }

    if variable_name not in session_vars:
        details["error"] = f"Variable '{variable_name}' not found in session variables."
        logger.warning(details["error"])
        return details

    obj = session_vars[variable_name]
    details["exists"] = True
    details["type"] = str(type(obj))

    try:
        # Length
        if hasattr(obj, "__len__"):
            try:
                details["length"] = len(obj)
            except TypeError: # Some objects have __len__ but it's not always applicable (e.g. Sized generator)
                pass


        # Pandas DataFrame
        if isinstance(obj, _pd.DataFrame):
            details["shape"] = list(obj.shape) # Convert tuple to list for JSON friendliness
            details["dtypes"] = {col: str(dtype) for col, dtype in obj.dtypes.items()}
            preview_df = obj.head(preview_elements)
            details["preview"] = preview_df.to_string(max_rows=preview_elements)
            if len(details["preview"]) > max_preview_str_len:
                details["preview"] = details["preview"][:max_preview_str_len] + "..."
            if details["length"] is None: # df.shape[0] is more reliable for row count
                details["length"] = obj.shape[0]


        # Pandas Series
        elif isinstance(obj, _pd.Series):
            details["shape"] = list(obj.shape) # Convert tuple to list
            details["dtypes"] = str(obj.dtype)
            preview_series = obj.head(preview_elements)
            details["preview"] = preview_series.to_string(max_rows=preview_elements)
            if len(details["preview"]) > max_preview_str_len:
                details["preview"] = details["preview"][:max_preview_str_len] + "..."
            if details["length"] is None:
                 details["length"] = len(obj)


        # NumPy Array
        elif isinstance(obj, _np.ndarray):
            details["shape"] = list(obj.shape) # Convert tuple to list
            details["dtypes"] = str(obj.dtype)
            # Previewing numpy arrays can be complex; show a slice
            if obj.ndim == 1:
                preview_arr = obj[:preview_elements]
            elif obj.ndim > 1:
                slicing = tuple([slice(None, preview_elements)] * obj.ndim)
                preview_arr = obj[slicing]
            else: # 0-dim array
                preview_arr = obj
            details["preview"] = _np.array_str(preview_arr, max_line_width=80, precision=3)
            if len(details["preview"]) > max_preview_str_len:
                details["preview"] = details["preview"][:max_preview_str_len] + "..."

        # List or Tuple
        elif isinstance(obj, (list, tuple)):
            if details["length"] is not None and details["length"] > preview_elements:
                preview_data = obj[:preview_elements]
                details["preview"] = repr(preview_data)[:-1] + f", ... ({details['length'] - preview_elements} more items)]" if isinstance(obj, list) else repr(preview_data)[:-1] + f", ... ({details['length'] - preview_elements} more items))"
            else:
                details["preview"] = repr(obj)
            if len(details["preview"]) > max_preview_str_len:
                details["preview"] = details["preview"][:max_preview_str_len] + "..."

        # Dictionary or Set
        elif isinstance(obj, (dict, set)):
            # For dicts/sets, previewing is a bit trickier to keep concise
            # We'll show a few items
            preview_items = []
            count = 0
            for item in (obj.items() if isinstance(obj, dict) else obj):
                if count >= preview_elements:
                    break
                preview_items.append(item)
                count += 1
            
            base_repr = repr(dict(preview_items) if isinstance(obj, dict) else set(preview_items))
            if details["length"] is not None and details["length"] > preview_elements:
                details["preview"] = base_repr[:-1] + f", ... ({details['length'] - preview_elements} more items)}}" if isinstance(obj,dict) else base_repr[:-1] + f", ... ({details['length'] - preview_elements} more items)}}"

            else:
                details["preview"] = base_repr
            if len(details["preview"]) > max_preview_str_len:
                details["preview"] = details["preview"][:max_preview_str_len] + "..."

        # Other types
        else:
            details["preview"] = repr(obj)
            if len(details["preview"]) > max_preview_str_len:
                details["preview"] = details["preview"][:max_preview_str_len] + "..."

    except Exception as e:
        error_msg = f"Error during inspection of '{variable_name}': {type(e).__name__}: {e}"
        details["error"] = error_msg
        logger.exception(f"! inspect_session_variable raised {error_msg}")
        # Keep whatever preview we might have gotten, or set to error
        if details["preview"] == "N/A" or not details["preview"]:
             details["preview"] = error_msg


    return details


# -------------------------------------------------------------------
# Utility to preview text files (avoids async in `py`)
# -------------------------------------------------------------------

@app.tool()
@log_tool
def peek_file(file_path: str, n_lines: int = 10) -> List[str]:
    """Return the first *n_lines* of a text file as a list of strings.

    This is useful for quickly inspecting a CSV or log file without
    resorting to asynchronous I/O inside the ``py`` REPL.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return [next(f).rstrip("\n") for _ in range(n_lines)]


# -------------------------------------------------------------------
# Feature extraction – concatenate semantic features
# -------------------------------------------------------------------


@app.tool()
@log_tool
def extract_log_features(
    log_vectors: Any,
    attributes_encoded: Any,
    timestamps: Any,
    max_feature_len: int = 100,
    *,
    save_as: str | None = None,
) -> Any:
    """Build a unified feature representation from logs and attributes.

    Parameters
    ----------
    log_vectors : numpy.ndarray | SessionVar
        Numerical vectors produced by :pyfunc:`vectorize_log_data`.
    attributes_encoded : numpy.ndarray | SessionVar
        Encoded categorical attributes from :pyfunc:`encode_log_attributes`.
    timestamps : pandas.Series | list | numpy.ndarray | SessionVar
        Timestamps associated with the log lines.  A list/array will be
        converted to :class:`pandas.Series` automatically.
    max_feature_len : int, default 100
        ``FeatureExtractorConfig.max_feature_len`` parameter.
    save_as : str, optional
        Store the resulting feature matrix under this key in
        *session_vars*.

    Returns
    -------
    Any
        Typically a ``numpy.ndarray`` combining all feature columns.
    """

    import numpy as _np
    import pandas as _pd

    # ------------------------------------------------------------------
    # Ensure the data structures align with FeatureExtractor expectations
    # (pandas Series/DataFrame instead of raw numpy arrays)
    # ------------------------------------------------------------------

    if not isinstance(log_vectors, _pd.Series):
        log_vectors_series = _pd.Series(log_vectors)
    else:
        log_vectors_series = log_vectors

    if not isinstance(attributes_encoded, _pd.DataFrame):
        attrs_df = _pd.DataFrame(attributes_encoded)
    else:
        attrs_df = attributes_encoded

    if not isinstance(timestamps, _pd.Series):
        timestamps_series = _pd.Series(timestamps)
    else:
        timestamps_series = timestamps

    feat_cfg = FeatureExtractorConfig(max_feature_len=max_feature_len)
    extractor = FeatureExtractor(feat_cfg)

    # The FeatureExtractor returns (metadata, feature_vector)
    _, feature_vector = extractor.convert_to_feature_vector(
        log_vectors_series, attrs_df, timestamps_series
    )

    if save_as:
        session_vars[save_as] = feature_vector

    return feature_vector


# -------------------------------------------------------------------
# Clustering – group log entries via algorithms such as K-Means
# -------------------------------------------------------------------


@app.tool()
@log_tool
def cluster_log_features(
    feature_vector: Any,
    algo_name: str = "kmeans",
    n_clusters: int = 7,
    clustering_params: dict | None = None,
    *,
    save_as: str | None = None,
) -> Any:
    """Cluster log entries based on their feature representation.

    Parameters
    ----------
    feature_vector : numpy.ndarray | pandas.DataFrame | SessionVar
        Feature matrix returned by :pyfunc:`extract_log_features`.
    algo_name : str, default "kmeans"
        Clustering algorithm name supported by LogAI.
    n_clusters : int, default 7
        Number of clusters for K-Means (ignored for other algos unless
        included in *clustering_params*).
    clustering_params : dict, optional
        Additional parameters forwarded to the algorithm-specific config
        object (e.g., ``KMeansParams``).
    save_as : str, optional
        Persist the resulting cluster ID Series for later use.

    Returns
    -------
    pandas.Series
        Series of *cluster_id* strings aligned with the input rows.
    """

    import numpy as _np
    import pandas as _pd

    feature_vector = _resolve(feature_vector)

    # Ensure DataFrame format as expected by LogAI's Clustering API
    if isinstance(feature_vector, _pd.DataFrame):
        features_df = feature_vector
    elif isinstance(feature_vector, _np.ndarray) and \
         feature_vector.ndim == 1 and \
         feature_vector.dtype == object and \
         len(feature_vector) > 0 and \
         hasattr(feature_vector[0], '__iter__') and \
         not isinstance(feature_vector[0], str):
        # Handle 1D numpy object array where each element is a list/array (feature vector)
        try:
            list_of_lists = [list(item) for item in feature_vector]
            features_df = _pd.DataFrame(list_of_lists)
            logger.info("Converted 1D object ndarray of iterables to DataFrame for clustering.")
        except Exception as e:
            logger.error(f"Failed to convert 1D object ndarray of iterables for clustering: {e}. Falling back.")
            # Fallback to direct conversion, which might cause the original error
            features_df = _pd.DataFrame(feature_vector)
    else:
        # Covers 2D numeric np.ndarray, list of lists, etc., where pd.DataFrame() usually works.
        features_df = _pd.DataFrame(feature_vector)

    clustering_params = clustering_params or {}

    # Build algorithm-specific params
    if algo_name == "kmeans":
        # Ensure n_clusters is set unless caller supplied it explicitly
        # Default to 'lloyd' algorithm for KMeans if not specified by the user
        # to avoid issues with 'auto' in some sklearn versions/LogAI.
        default_kmeans_algo_params = {"n_clusters": n_clusters, "algorithm": "lloyd"}
        clustering_params = {**default_kmeans_algo_params, **clustering_params}
        algo_params = KMeansParams(**clustering_params)
    else:
        # For other algorithms pass raw dict if provided
        algo_params = clustering_params  # type: ignore[assignment]

    cfg = ClusteringConfig(algo_name=algo_name, algo_params=algo_params)
    clusterer = Clustering(cfg)

    clusterer.fit(features_df)
    cluster_ids = clusterer.predict(features_df)

    cluster_series = _pd.Series(cluster_ids.astype(str), name="cluster_id")

    if save_as:
        session_vars[save_as] = cluster_series

    return cluster_series


@app.tool()
@log_tool
def combine_csv_files(
    input_source: Union[str, List[str]], # Changed from file_paths
    file_glob_pattern: str = "*.csv",   # New parameter
    sort_by_columns: Union[List[str], str, None] = None,
    ascending: Union[bool, List[bool]] = True,
    join_axes_how: str = 'outer',
    drop_columns: List[str] | None = None,
    *,
    save_as: str | None = None
) -> Any: 
    """
    Reads multiple CSV files, concatenates them into a single pandas DataFrame,
    optionally sorts the result, and optionally drops specified columns.

    This tool can source CSV files in two ways:
    1. By providing a directory path: If `input_source` is a string representing a
       directory path, the tool searches within this directory for files matching
       the `file_glob_pattern` (defaulting to '*.csv').
    2. By providing an explicit list: If `input_source` is a list of strings, each
       string is treated as either a direct file path or the name of a session
       variable that resolves to a file path or a list of file paths.

    Once the list of CSV files is determined, each is read into a pandas DataFrame.
    These DataFrames are then concatenated row-wise (stacked vertically).

    The `join_axes_how` parameter dictates how columns are handled if they differ
    across the input CSVs:
    - 'outer' (default): Includes all columns present in any file. Columns not
      present in a particular file will have NaN values for rows originating
      from that file.
    - 'inner': Includes only columns that are common to all input files. Rows from
      files that have additional columns will only include the common ones.

    After concatenation, specified columns can be dropped using `drop_columns`.
    Finally, the resulting DataFrame can be sorted based on one or more columns
    as specified by `sort_by_columns` and `ascending`.

    The final DataFrame can be saved into the session_vars dictionary for use
    by subsequent tools if a `save_as` name is provided.

    Parameters
    ----------
    input_source : Union[str, List[str]] | SessionVar
        The source for CSV files. Can be:
        1. A string representing a path to a directory. Files within this directory
           matching `file_glob_pattern` will be processed.
        2. A list of strings, where each string is either:
           a. A direct path (absolute or relative) to a CSV file.
           b. The name of a session variable. This variable should resolve to
              either a single string path or a list of string paths to CSV files.
           Nested lists of paths from session variables will be flattened.
        If `input_source` itself is a session variable name, it should resolve to
        either a directory path (string) or a list of file path strings.
    file_glob_pattern : str, default "*.csv"
        A glob pattern used to match files when `input_source` is a directory path.
        This parameter is ignored if `input_source` is a list.
        Examples: "*.csv", "data_prefix_*.csv", "results??.txt".
    sort_by_columns : Union[List[str], str], optional
        The name of a single column (as a string) or a list of column names
        (as a list of strings) by which to sort the combined DataFrame.
        If `None` (default), no sorting is performed.
        The columns specified must exist in the combined DataFrame.
    ascending : Union[bool, List[bool]], default True
        Determines the sort order for the columns specified in `sort_by_columns`.
        - If a single boolean value, all columns in `sort_by_columns` are sorted
          in that specified order.
        - If a list of boolean values, it must have the same length as
          `sort_by_columns`, specifying the order for each corresponding column.
    join_axes_how : {'outer', 'inner'}, default 'outer'
        The method for joining axes (columns) during concatenation (`pandas.concat` join parameter).
        - 'outer': Union of all columns, fills missing values with NaN.
        - 'inner': Intersection of columns, only common columns are kept.
    drop_columns : List[str], optional
        A list of column names to remove from the combined DataFrame after concatenation.
        Non-existent columns in this list are ignored without error.
    save_as : str, optional
        If provided, the resulting pandas DataFrame is stored in `session_vars`
        with this name as the key. If `None`, the DataFrame is returned but not stored.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame representing the combined data from all sourced CSV files,
        after optional column dropping and sorting.
        (Note: FastMCP tools are typically typed `Any` in the framework).

    Raises
    ------
    ValueError
        - If `input_source` (after resolving if it's a session variable) is neither
          a string (directory path) nor a list of strings.
        - If `input_source` is a directory path that does not exist or is not a directory.
        - If no CSV files are found or resolved from `input_source`.
        - If `join_axes_how` is not 'outer' or 'inner'.
        - If `sort_by_columns` specifies non-existent columns.
        - If an error occurs during CSV reading/parsing (other than `FileNotFoundError`).
    FileNotFoundError
        If an explicit file path in `input_source` (when it's a list) does not exist.
        (Note: `glob` does not raise FileNotFoundError for non-matching patterns in an existing directory).
    TypeError
        - If `sort_by_columns` or items in `drop_columns` are not strings.
        - If `save_as` is provided but is not a string.

    Examples
    --------
    >>> # Scenario 1: Combine all CSVs from a directory './my_logs/'
    >>> # Assume ./my_logs/log_part1.csv and ./my_logs/log_part2.csv exist.
    >>> combine_csv_files(
    ...     input_source="./my_logs",
    ...     file_glob_pattern="log_*.csv",
    ...     sort_by_columns="timestamp",
    ...     save_as="all_my_logs"
    ... )

    >>> # Scenario 2: Combine specific CSV files and a session variable list
    >>> # session_vars["archive_files"] = ["archive/old_log1.csv", "archive/old_log2.csv"]
    >>> combine_csv_files(
    ...     input_source=["current_day.csv", "archive_files"],
    ...     join_axes_how='inner',
    ...     save_as="combined_current_archive"
    ... )
    """
    import pandas as _pd
    import glob
    import os

    # Resolve input_source if it's a session variable name itself
    resolved_input_source = _resolve(input_source)

    actual_file_paths: List[str] = []

    if isinstance(resolved_input_source, str): # Interpreted as a directory path
        directory_path = resolved_input_source
        if not os.path.isdir(directory_path):
            raise ValueError(f"Input source '{directory_path}' is a string but not a valid directory.")
        
        glob_path = os.path.join(directory_path, file_glob_pattern)
        logger.info(f"Searching for files in directory '{directory_path}' with pattern '{file_glob_pattern}' (glob path: '{glob_path}').")
        found_files = glob.glob(glob_path)
        if not found_files:
            logger.warning(f"No files found in directory '{directory_path}' matching pattern '{file_glob_pattern}'.")
            # Return an empty DataFrame or raise error? Current logic will raise later if no DFs to concat.
            # For consistency, let it flow, and if dataframes_to_concat is empty, it will raise.
        else:
            logger.info(f"Found {len(found_files)} file(s): {found_files}")    
        actual_file_paths.extend(sorted(found_files)) # Sort for deterministic order

    elif isinstance(resolved_input_source, list):
        logger.info(f"Processing input_source as a list of file paths/session variables: {resolved_input_source}")
        for path_or_var_or_list_item in resolved_input_source:
            resolved_list_item = _resolve(path_or_var_or_list_item)
            if isinstance(resolved_list_item, list):
                for sub_path_item in resolved_list_item:
                    sub_path = _resolve(sub_path_item)
                    if not isinstance(sub_path, str):
                        raise ValueError(
                            f"Item '{sub_path_item}' within a nested list resolved to an unexpected type: {type(sub_path)}. Expected str."
                        )
                    actual_file_paths.append(sub_path)
            elif isinstance(resolved_list_item, str):
                actual_file_paths.append(resolved_list_item)
            else:
                raise ValueError(
                    f"Item '{path_or_var_or_list_item}' in input_source list resolved to an unexpected type: {type(resolved_list_item)}. Expected str or list of str."
                )
    else:
        raise ValueError(
            f"'input_source' (after resolving '{input_source}') must be a directory path (str) or a list of file paths/session vars. Got: {type(resolved_input_source)}."
        )

    if not actual_file_paths:
        # This handles case where directory was valid but no files matched, or list was empty/resolved to empty.
        raise ValueError("No CSV files found or resolved from the input_source to process.")

    if join_axes_how not in ['outer', 'inner']:
        raise ValueError("`join_axes_how` must be either 'outer' or 'inner'.")
    
    valid_join_how: Literal['inner', 'outer'] = join_axes_how # type: ignore

    dataframes_to_concat: List[_pd.DataFrame] = []
    for file_path_str in actual_file_paths:
        try:
            logger.info(f"Reading CSV file: {file_path_str}")
            df = _pd.read_csv(file_path_str)
            if df.empty and file_path_str: # Log if a non-empty path results in an empty df
                 logger.warning(f"CSV file '{file_path_str}' was read as an empty DataFrame.")
            dataframes_to_concat.append(df)
        except FileNotFoundError:
            logger.error(f"File not found during explicit read: {file_path_str}") # glob.glob won't find it, but explicit paths from list might fail here
            raise
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path_str}: {type(e).__name__}: {e}")
            raise ValueError(f"Failed to read or parse CSV file '{file_path_str}'. Error: {type(e).__name__}: {e}") from e

    if not dataframes_to_concat: # Should be caught by the earlier check on actual_file_paths, but as a safeguard.
        raise ValueError("No DataFrames were loaded to concatenate. This implies all resolved file paths were empty or unreadable after initial discovery.")

    logger.info(f"Concatenating {len(dataframes_to_concat)} DataFrame(s) with join_axes_how='{valid_join_how}'.")
    combined_df = _pd.concat(dataframes_to_concat, join=valid_join_how, ignore_index=True)

    if combined_df.empty and dataframes_to_concat:
        logger.warning("Concatenation resulted in an empty DataFrame. Check CSV contents and 'join_axes_how'.")

    if drop_columns:
        if not isinstance(drop_columns, list) or not all(isinstance(col, str) for col in drop_columns):
            raise TypeError("`drop_columns` must be a list of strings.")
        
        columns_present_for_dropping = [col for col in drop_columns if col in combined_df.columns]
        columns_not_found_for_dropping = [col for col in drop_columns if col not in combined_df.columns]

        if columns_not_found_for_dropping:
            logger.warning(
                f"Columns specified in 'drop_columns' not found and will be ignored: {columns_not_found_for_dropping}"
            )
        
        if columns_present_for_dropping:
            logger.info(f"Dropping columns: {columns_present_for_dropping}")
            combined_df = combined_df.drop(columns=columns_present_for_dropping)
        else:
            logger.info("No columns specified in 'drop_columns' were found to drop.")

    if sort_by_columns:
        actual_sort_cols: List[str]
        if isinstance(sort_by_columns, str):
            actual_sort_cols = [sort_by_columns]
        elif isinstance(sort_by_columns, list) and all(isinstance(col, str) for col in sort_by_columns):
            actual_sort_cols = sort_by_columns
        else:
            raise TypeError("`sort_by_columns` must be a string or a list of strings.")

        missing_sort_cols = [col for col in actual_sort_cols if col not in combined_df.columns]
        if missing_sort_cols:
            err_msg = (
                f"Cannot sort by columns not present in DataFrame: {missing_sort_cols}. "
                f"Available columns: {combined_df.columns.tolist()}"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        
        try:
            logger.info(f"Sorting DataFrame by columns: {actual_sort_cols}, ascending: {ascending}")
            combined_df = combined_df.sort_values(by=actual_sort_cols, ascending=ascending, ignore_index=True)
        except Exception as e:
            logger.error(f"Error sorting DataFrame by {actual_sort_cols}: {type(e).__name__}: {e}")
            raise ValueError(f"Failed to sort DataFrame. Error: {type(e).__name__}: {e}") from e

    if save_as:
        if not isinstance(save_as, str):
            raise TypeError("`save_as` parameter must be a string.")
        logger.info(f"Saving combined DataFrame to session variable: '{save_as}'")
        session_vars[save_as] = combined_df

    return combined_df


@app.tool()
@log_tool
def load_openset_log_data(dataset_name: str, data_dir: str = "../datasets", *, save_as: str | None = None) -> Any:
    """Load a standard open dataset provided by LogAI by its name.

    This tool simplifies loading common log datasets that are part of LogAI's
    openset collection (e.g., HDFS, BGL, HealthApp). It uses LogAI's
    `OpenSetDataLoader` which is designed to handle the specifics of these
    datasets, such as their file structure and initial parsing hints if any.

    The tool constructs an expected file path based on the `dataset_name` and
    `data_dir`. For most datasets, it expects `data_dir/dataset_name/dataset_name.log`.
    For "HealthApp", it specifically looks for `HealthApp_2000.log` which is a common
    sample size for that dataset. The `data_dir` should point to the parent directory
    containing individual dataset folders.

    The loaded data is returned as a LogAI `LogRecordObject`, which encapsulates
    various aspects of the log data (timestamps, body, attributes, etc.). This
    object can then be passed to other LogAI-specific tools in the MCP server.

    Parameters
    ----------
    dataset_name : str
        The official name of the open dataset to load, as recognized by LogAI's
        `OpenSetDataLoader`. Common examples include "HealthApp", "HDFS", "BGL",
        "Thunderbird", "Linux", etc. This name is crucial as LogAI might use it
        internally to apply dataset-specific loading configurations.
    data_dir : str, default "../datasets"
        The base directory where the subdirectories for each openset dataset are
        located. For instance, if loading "HealthApp", and its log file is at
        `../datasets/HealthApp/HealthApp_2000.log`, then `data_dir` should be
        `"../datasets"`. The path can be relative to the server's current
        working directory or an absolute path.
    save_as : str, optional
        If a string name is provided, the resulting LogAI `LogRecordObject` is
        stored in the `session_vars` dictionary with this name as the key.
        This allows the loaded data to be referenced by this name in subsequent
        tool calls. If `None` (default), the object is returned but not stored
        in the session.

    Returns
    -------
    Any
        A LogAI `LogRecordObject` (or a similar Dataset-like wrapper object
        provided by LogAI) containing the loaded log data. The exact structure
        can be introspected using tools like `describe_log_record` or
        `inspect_session_variable`.

    Raises
    ------
    FileNotFoundError
        If the constructed file path for the specified `dataset_name` and
        `data_dir` does not exist on the filesystem. The error message will
        indicate the path that was tried.
    RuntimeError
        If LogAI's `OpenSetDataLoader` encounters an error during the loading
        process for reasons other than a simple file not found (e.g., issues
        with LogAI's internal configuration for the dataset, permission errors
        not caught as FileNotFoundError, or other exceptions raised by LogAI).
        The original exception from LogAI will be chained.
    TypeError
        If `save_as` is provided but is not a string.

    Examples
    --------
    >>> # Scenario 1: Load HealthApp dataset and save it
    >>> load_openset_log_data(dataset_name="HealthApp", save_as="healthapp_data")
    >>> # Now, session_vars["healthapp_data"] contains the LogRecordObject.
    >>> # You can then inspect it:
    >>> # describe_log_record("healthapp_data")

    >>> # Scenario 2: Load HDFS dataset with a custom data directory
    >>> load_openset_log_data(
    ...     dataset_name="HDFS",
    ...     data_dir="/mnt/shared_log_datasets",
    ...     save_as="hdfs_logs"
    ... )
    >>> # session_vars["hdfs_logs"] contains the HDFS LogRecordObject.

    Notes
    -----
    - The successful execution of this tool depends on the correct installation
      of LogAI and the availability of the specified open datasets in the expected
      directory structure.
    - The `OpenSetDataLoader` might perform some initial processing or apply
      default configurations specific to the dataset being loaded.
    - It's recommended to use `os.path.abspath` or ensure `data_dir` is robustly
      defined if the server's working directory can vary. This implementation
      now uses `os.path.abspath` on the constructed file path.
    """
    import os # Local import for clarity
    from logai.dataloader.openset_data_loader import OpenSetDataLoader, OpenSetDataLoaderConfig # Ensure LogAI specific imports

    if dataset_name == "HealthApp":
        filename = f"{dataset_name}_2000.log"
    else:
        filename = f"{dataset_name}.log"

    full_filepath = os.path.join(data_dir, dataset_name, filename)
    full_filepath = os.path.abspath(full_filepath)

    logger.info(f"Attempting to load OpenSet dataset '{dataset_name}' from: {full_filepath}")

    if not os.path.exists(full_filepath):
        error_msg = (
            f"Constructed filepath for dataset '{dataset_name}' not found: {full_filepath}. "
            f"Ensure dataset exists or adjust 'data_dir'."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        config = OpenSetDataLoaderConfig(
            dataset_name=dataset_name,
            filepath=full_filepath
        )
        loader = OpenSetDataLoader(config)
        logrecord = loader.load_data()
    except Exception as e:
        logger.exception(f"Error loading OpenSet dataset '{dataset_name}' using LogAI: {e}")
        raise RuntimeError(f"LogAI OpenSetDataLoader failed for '{dataset_name}': {type(e).__name__}: {e}") from e

    if save_as:
        session_vars[save_as] = logrecord
    return logrecord


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
    save_as: str | None = None
) -> Any:
    """
    Generates counter vectors from parsed logs, attributes, and timestamps,
    suitable for time-series anomaly detection. This tool wraps LogAI's
    `FeatureExtractor.convert_to_counter_vector` method.

    Counter vectors represent logs as time series where each point in time shows
    the count of occurrences of log events (or patterns) optionally grouped by
    certain categories. This is a common feature engineering step for applying
    time-series based anomaly detection algorithms.

    The tool requires:
    - Parsed log lines (log templates).
    - Corresponding attributes (structured data associated with each log line).
    - Timestamps for each log line.

    It groups these logs based on the specified time window (`group_by_time`)
    and categories (`group_by_category`) to create multivariate or univariate
    time series of log counts.

    Parameters
    ----------
    parsed_loglines : pd.Series | SessionVar
        A pandas Series containing the parsed log lines (log templates). This is
        typically the output from a tool like `parse_log_data` (specifically,
        the 'parsed_logline' component). It can be provided directly or as a
        name of a session variable.
    attributes : pd.DataFrame | SessionVar
        A pandas DataFrame containing attributes associated with each log line.
        This usually comes from the `LogRecordObject` loaded by tools like
        `load_file_log_data` or `load_openset_log_data`, or from the output
        of `preprocess_log_data`.
    timestamps : pd.Series | SessionVar
        A pandas Series of timestamps corresponding to each log line. This also
        typically originates from the loaded `LogRecordObject`.
    group_by_time : str
        A time window string (e.g., "5min", "1H", "1D") used to aggregate log
        counts into time buckets. This defines the sampling frequency of the
        resulting time series.
    group_by_category : List[str]
        A list of column names from the `attributes` DataFrame and/or the special
        string 'parsed_logline'. The log counts will be aggregated distinctly
        for each unique combination of values in these categories. For example,
        if `group_by_category=['parsed_logline', 'severity']`, a separate time
        series of counts will be generated for each unique log template and
        severity level combination.
    feature_extractor_params : dict, optional
        A dictionary of additional keyword arguments to be passed to LogAI's
        `FeatureExtractorConfig`. This allows for fine-tuning other aspects
        of the feature extraction process if supported by LogAI.
        If `None` (default), only `group_by_time` and `group_by_category` are used.
    save_as : str, optional
        If a string name is provided, the resulting counter vector (a pandas
        DataFrame) is stored in the `session_vars` dictionary with this name.
        If `None` (default), the DataFrame is returned but not stored in the session.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame representing the generated counter vector. The structure
        of this DataFrame typically includes:
        - Timestamp columns (e.g., `logai.utils.constants.LOG_TIMESTAMPS`).
        - Count columns (e.g., `logai.utils.constants.LOG_COUNTS`).
        - Columns corresponding to the categories specified in `group_by_category`,
          identifying each unique time series.
        (Note: FastMCP tools are typically typed to return `Any` in the framework).

    Raises
    ------
    TypeError
        - If `parsed_loglines` does not resolve to a pandas Series.
        - If `attributes` does not resolve to a pandas DataFrame.
        - If `timestamps` does not resolve to a pandas Series.
        - If `save_as` is provided but is not a string.
    Exception
        Any exception raised by LogAI's `FeatureExtractor` during the
        `convert_to_counter_vector` process (e.g., issues with configuration,
        data type mismatches not caught by initial checks, problems during
        aggregation).

    Examples
    --------
    >>> # Assume session_vars has:
    >>> # 'parsed_logs_series': a pd.Series of parsed log templates
    >>> # 'log_attributes_df': a pd.DataFrame of log attributes
    >>> # 'log_timestamps_series': a pd.Series of timestamps
    >>>
    >>> extract_timeseries_features(
    ...     parsed_loglines="parsed_logs_series",
    ...     attributes="log_attributes_df",
    ...     timestamps="log_timestamps_series",
    ...     group_by_time="10m",
    ...     group_by_category=['parsed_logline', 'hostname'],
    ...     save_as="log_counter_vectors"
    ... )
    >>> # session_vars["log_counter_vectors"] will now hold the DataFrame of counter vectors.
    >>> # This can then be used by `detect_timeseries_anomalies`.

    Notes
    -----
    - The input pandas Series and DataFrame (`parsed_loglines`, `attributes`, `timestamps`)
      must be aligned (i.e., have the same length and correspond row by row).
    - The column names used in `group_by_category` must exist in the `attributes`
      DataFrame or be 'parsed_logline'.
    - LogAI's `FeatureExtractor` handles the underlying complexities of time-based
      aggregation and categorical grouping.
    """
    import pandas as _pd # Local import
    from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor # LogAI import

    _parsed_loglines = _resolve(parsed_loglines)
    _attributes = _resolve(attributes)
    _timestamps = _resolve(timestamps)

    # Basic type validation
    if not isinstance(_parsed_loglines, _pd.Series):
        raise TypeError(f"Expected 'parsed_loglines' to be a pandas Series, got {type(_parsed_loglines)}")
    if not isinstance(_attributes, _pd.DataFrame):
        raise TypeError(f"Expected 'attributes' to be a pandas DataFrame, got {type(_attributes)}")
    if not isinstance(_timestamps, _pd.Series):
        raise TypeError(f"Expected 'timestamps' to be a pandas Series, got {type(_timestamps)}")

    config_params = {
        "group_by_time": group_by_time,
        "group_by_category": group_by_category,
        **(feature_extractor_params or {})
    }
    config = FeatureExtractorConfig(**config_params)
    feature_extractor = FeatureExtractor(config)

    logger.info(f"Extracting timeseries features with config: {config_params}")
    counter_vector_df = feature_extractor.convert_to_counter_vector(
        log_pattern=_parsed_loglines,
        attributes=_attributes,
        timestamps=_timestamps
    )

    if save_as:
        session_vars[save_as] = counter_vector_df
    return counter_vector_df


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
    min_timeseries_length: int = 10, 
    *,
    save_scores_as: str | None = None,
    save_anomalies_as: str | None = None 
) -> Dict[str, Any]:
    """
    Detects anomalies in time-series data, typically counter vectors generated
    by `extract_timeseries_features`. Wraps LogAI's `AnomalyDetector`.

    This tool can process multiple time series if they are grouped within the
    input `counter_vector_data` DataFrame. If `attribute_group_cols` are specified,
    it iterates through each unique combination of these attributes, treating each
    as an independent time series. For each series, it performs a train/test split
    (chronologically), fits the specified anomaly detection algorithm, and then
    predicts anomaly scores on the test portion.

    The results, including anomaly scores and optionally the original data points
    flagged as anomalous, can be saved to session variables.

    Parameters
    ----------
    counter_vector_data : pd.DataFrame | SessionVar
        A pandas DataFrame containing the time-series data (counter vectors).
        This is typically the output from `extract_timeseries_features`.
        It must contain at least a timestamp column and a count/value column.
    algo_name : str
        The name of the anomaly detection algorithm to use, as supported by
        LogAI's `AnomalyDetector` (e.g., 'dbl', 'arima', 'ets', from Merlion).
    timestamp_col : str
        The name of the column in `counter_vector_data` that contains the
        timestamps for the time series. This should align with LogAI's constants
        like `logai.utils.constants.LOG_TIMESTAMPS` if generated by LogAI tools.
    count_col : str
        The name of the column in `counter_vector_data` that contains the
        numerical values (e.g., log counts) of the time series to be analyzed.
        Should align with constants like `logai.utils.constants.LOG_COUNTS`.
    attribute_group_cols : List[str], optional
        A list of column names present in `counter_vector_data` that define
        unique groups, each forming an independent time series. For example, if
        `counter_vector_data` includes counts for different 'event_template' and
        'hostname', specifying `attribute_group_cols=['event_template', 'hostname']`
        will process anomalies for each unique template-host pair separately.
        If `None` (default) or if specified columns are not found, the entire
        `counter_vector_data` is treated as a single time series.
    anomaly_detection_params : dict, optional
        A dictionary of additional parameters to be passed to the specific
        anomaly detection algorithm's configuration (e.g., parameters for
        Merlion's DynamicBaseline (dbl) algorithm if `algo_name='dbl'`).
        These are typically used for `algo_params` in LogAI's `AnomalyDetectionConfig`.
    train_split_ratio : float, default 0.7
        The proportion of data to use for training the anomaly detection model for
        each time series. The split is done chronologically; the first
        `train_split_ratio` fraction of data points forms the training set, and
        the remainder forms the test set. Must be between 0 and 1 (exclusive).
    min_timeseries_length : int, default 10
        The minimum number of data points a time series (or a group's time series)
        must have to be considered for anomaly detection. Series shorter than this
        will be skipped. This also affects the train/test split feasibility.
    save_scores_as : str, optional
        If a string name is provided, a pandas DataFrame containing the anomaly
        scores for the test portions of all processed series is stored in
        `session_vars` under this name. The DataFrame will have the original
        index from `counter_vector_data` to allow alignment.
    save_anomalies_as : str, optional
        If a string name is provided, a pandas DataFrame containing the subset of
        rows from the original `counter_vector_data` that were flagged as
        anomalous (based on a positive anomaly score) is stored in `session_vars`
        under this name.

    Returns
    -------
    dict
        A dictionary summarizing the execution and results:
        - 'processed_series_count': Number of individual time series processed.
        - 'anomalies_found_count': Total number of data points flagged as anomalous
          (score > 0) across all processed test sets.
        - 'anomaly_scores_var': The `save_scores_as` name if scores were saved, else `None`.
        - 'anomalous_data_var': The `save_anomalies_as` name if anomalous data was saved, else `None`.

    Raises
    ------
    TypeError
        - If `counter_vector_data` does not resolve to a pandas DataFrame.
        - If `save_scores_as` or `save_anomalies_as` are provided but are not strings.
    ValueError
        - If `train_split_ratio` is not between 0 and 1 (exclusive).
        - If required `timestamp_col` or `count_col` are not found in the data for a series.
    Exception
        Any exception raised by LogAI's `AnomalyDetector` during model fitting
        or prediction (e.g., issues with algorithm configuration, data suitability
        for the chosen algorithm, numerical errors).

    Examples
    --------
    >>> # Assume 'log_counter_vectors' (from extract_timeseries_features) is in session_vars.
    >>> # It has columns: 'timestamp', 'event_count', 'parsed_logline', 'hostname'.
    >>> detect_timeseries_anomalies(
    ...     counter_vector_data="log_counter_vectors",
    ...     algo_name="dbl", # Dynamic Baseline algorithm
    ...     timestamp_col="timestamp", # Adjust if your column name is different
    ...     count_col="event_count",   # Adjust if your column name is different
    ...     attribute_group_cols=['parsed_logline', 'hostname'],
    ...     train_split_ratio=0.8,
    ...     save_scores_as="ts_anomaly_scores",
    ...     save_anomalies_as="ts_anomalous_points"
    ... )
    >>> # Results summary is returned.
    >>> # session_vars["ts_anomaly_scores"] contains DataFrame of scores.
    >>> # session_vars["ts_anomalous_points"] contains DataFrame of anomalous counter vector rows.

    Notes
    -----
    - The definition of an anomaly (e.g., score > 0) is based on common LogAI/Merlion conventions
      but might need adjustment or configuration depending on the specific `algo_name` used.
    - The input `counter_vector_data` should ideally be sorted by time if it's treated as a single
      series, or within each group if `attribute_group_cols` are used. This tool explicitly sorts
      each processed series by `timestamp_col` before splitting.
    - LogAI's `AnomalyDetector` often leverages algorithms from the Merlion library.
    """
    import pandas as _pd 
    from sklearn.model_selection import train_test_split
    from logai.analysis.anomaly_detector import AnomalyDetector, AnomalyDetectionConfig
    from logai.utils import constants as logai_constants 

    _counter_vector_df = _resolve(counter_vector_data)
    if not isinstance(_counter_vector_df, _pd.DataFrame):
        raise TypeError(f"Expected 'counter_vector_data' to be pandas DataFrame, got {type(_counter_vector_df)}")

    if not (0 < train_split_ratio < 1):
        raise ValueError("train_split_ratio must be between 0 and 1 (exclusive).")

    ad_config = AnomalyDetectionConfig(
        algo_name=algo_name,
        algo_params=anomaly_detection_params or {}
    )
    
    all_anomaly_scores = []
    processed_series_count = 0
    total_anomalies_found = 0

    if attribute_group_cols and all(col in _counter_vector_df.columns for col in attribute_group_cols):
        _counter_vector_df['_temp_group_key'] = _counter_vector_df[attribute_group_cols].astype(str).agg('-'.join, axis=1)
        unique_groups = _counter_vector_df['_temp_group_key'].unique()
        logger.info(f"Processing {len(unique_groups)} unique time series based on {attribute_group_cols}.")
    else:
        if attribute_group_cols: 
             logger.warning(f"Attribute group columns {attribute_group_cols} not all found. Treating as single time series.")
        _counter_vector_df['_temp_group_key'] = "single_series"
        unique_groups = ["single_series"]
        logger.info("No valid attribute_group_cols, treating as a single time series.")

    for group_key in unique_groups:
        series_df = _counter_vector_df[_counter_vector_df['_temp_group_key'] == group_key].copy()
        series_df.sort_values(by=timestamp_col, inplace=True) 

        if len(series_df) < min_timeseries_length:
            logger.warning(f"Skipping series for group '{group_key}': length {len(series_df)} < min_timeseries_length {min_timeseries_length}.")
            continue
        
        if timestamp_col not in series_df.columns or count_col not in series_df.columns:
            logger.error(f"Timestamp column '{timestamp_col}' or count column '{count_col}' not found in series for group '{group_key}'. Skipping.")
            continue

        data_for_splitting = series_df[[timestamp_col, count_col]]
        
        train_size = int(len(data_for_splitting) * train_split_ratio)
        if train_size < 1 or (len(data_for_splitting) - train_size) < 1: 
            logger.warning(f"Series for group '{group_key}' is too short for train/test split ({len(data_for_splitting)} points, train_ratio {train_split_ratio}). Skipping.")
            continue
            
        train_df = data_for_splitting.iloc[:train_size]
        test_df = data_for_splitting.iloc[train_size:]

        logger.info(f"Processing series for group '{group_key}': Train size {len(train_df)}, Test size {len(test_df)}")

        try:
            anomaly_detector = AnomalyDetector(ad_config)
            anomaly_detector.fit(train_df) 
            anomaly_scores_series = anomaly_detector.predict(test_df) 
            
            scores_df = _pd.DataFrame({'anomaly_score': anomaly_scores_series.values}, index=test_df.index)
            all_anomaly_scores.append(scores_df)
            total_anomalies_found += (anomaly_scores_series > 0).sum() 
            processed_series_count += 1
        except Exception as e:
            logger.error(f"Error detecting anomalies for series group '{group_key}': {type(e).__name__} - {e}")
            continue 

    final_scores_df = _pd.DataFrame()
    if all_anomaly_scores:
        final_scores_df = _pd.concat(all_anomaly_scores).sort_index()
    
    if '_temp_group_key' in _counter_vector_df.columns:
        _counter_vector_df.drop(columns=['_temp_group_key'], inplace=True)


    results_summary = {
        "processed_series_count": processed_series_count,
        "anomalies_found_count": total_anomalies_found, 
        "anomaly_scores_var": None,
        "anomalous_data_var": None
    }

    if save_scores_as and not final_scores_df.empty:
        session_vars[save_scores_as] = final_scores_df
        results_summary["anomaly_scores_var"] = save_scores_as
        logger.info(f"Saved anomaly scores to session variable: {save_scores_as}")

    if save_anomalies_as and not final_scores_df.empty:
        anomalous_indices = final_scores_df[final_scores_df['anomaly_score'] > 0].index
        if not anomalous_indices.empty:
            anomalous_data = _counter_vector_df.loc[anomalous_indices].copy()
            session_vars[save_anomalies_as] = anomalous_data
            results_summary["anomalous_data_var"] = save_anomalies_as
            logger.info(f"Saved {len(anomalous_data)} anomalous data points to session variable: {save_anomalies_as}")
        else:
            logger.info("No anomalies found to save based on score > 0.")


    return results_summary

@app.tool()
@log_tool
def detect_semantic_anomalies(
    feature_vector_data: Any,
    algo_name: str, 
    anomaly_detection_params: Dict[str, Any] | None = None, 
    train_data_var: str | None = None, 
    test_data_var: str | None = None,  
    train_split_ratio: float | None = 0.7, 
    *,
    save_predictions_as: str | None = None,
    save_anomalous_indices_as: str | None = None 
) -> Dict[str, Any]:
    """
    Detects anomalies in log data based on semantic feature vectors, which are
    typically generated by `extract_log_features`. This tool wraps LogAI's
    `AnomalyDetector` for use with algorithms suitable for feature-based (non-time-series)
    anomaly detection, such as Isolation Forest, Local Outlier Factor (LOF), etc.

    The tool can operate in two modes for data splitting:
    1. Pre-split data: If `train_data_var` and `test_data_var` (session variable
       names pointing to feature vectors) are provided, these are used directly
       for training and testing.
    2. Automatic split: If pre-split data is not provided, and `train_split_ratio`
       is a float (e.g., 0.7), the input `feature_vector_data` is split into
       training and testing sets using `sklearn.model_selection.train_test_split`.
       If `train_split_ratio` is `None` (and no pre-split data), the model is
       trained on the entire `feature_vector_data`, and predictions are also made
       on the entire dataset.

    The raw predictions from the algorithm (e.g., -1 for anomaly, 1 for normal
    in scikit-learn's Isolation Forest) and the indices of data points flagged as
    anomalous can be saved to session variables.

    Parameters
    ----------
    feature_vector_data : pd.DataFrame | np.ndarray | SessionVar
        The primary dataset of semantic feature vectors. This is typically the
        output from `extract_log_features`. It can be a pandas DataFrame, a
        NumPy ndarray, or the name of a session variable holding such data.
        If `train_data_var` and `test_data_var` are not used, this data will be
        split or used whole for training and testing.
    algo_name : str
        The name of the anomaly detection algorithm to use, as supported by
        LogAI's `AnomalyDetector` for feature-based detection (e.g.,
        'isolation_forest', 'lof', 'one_class_svm').
    anomaly_detection_params : dict, optional
        A dictionary of parameters specific to the chosen `algo_name`. These are
        forwarded to LogAI's `AnomalyDetectionConfig` (often as `algo_params`).
        For example, for 'isolation_forest', this could include `n_estimators`,
        `max_samples`, etc. Refer to LogAI or scikit-learn documentation for
        the specific algorithm's parameters.
        Example: `{"n_estimators": 100, "contamination": "auto"}`
    train_data_var : str, optional
        The name of a session variable that holds the pre-split training feature
        vectors (as a DataFrame or ndarray). If provided along with `test_data_var`,
        `feature_vector_data` and `train_split_ratio` are ignored for splitting.
    test_data_var : str, optional
        The name of a session variable that holds the pre-split testing feature
        vectors (as a DataFrame or ndarray). Must be provided if `train_data_var`
        is provided.
    train_split_ratio : float, optional, default 0.7
        The proportion of `feature_vector_data` to allocate to the training set
        if pre-split data (via `train_data_var`, `test_data_var`) is not used.
        The value must be between 0.0 and 1.0. If `None`, and no pre-split data
        is provided, the model is trained on the full `feature_vector_data` and
        also tested on the full set. `shuffle=True` is used in `train_test_split`.
    save_predictions_as : str, optional
        If a string name is provided, the raw prediction results from the
        anomaly detection algorithm (typically a pandas Series where the index
        aligns with the test data) are stored in `session_vars` under this name.
        The values depend on the algorithm (e.g., for Isolation Forest, typically
        1 for inliers, -1 for outliers).
    save_anomalous_indices_as : str, optional
        If a string name is provided, a pandas Index object containing the original
        indices (from `feature_vector_data` or `test_data_var`) of data points
        that were flagged as anomalous is stored in `session_vars` under this name.
        The definition of "anomalous" depends on the `algo_name` (e.g., prediction == -1
        for scikit-learn's Isolation Forest, or prediction == 1 if LogAI wraps it differently).

    Returns
    -------
    dict
        A dictionary summarizing the execution and results:
        - 'predictions_var': The `save_predictions_as` name if predictions were saved, else `None`.
        - 'anomalous_indices_var': The `save_anomalous_indices_as` name if anomalous
          indices were saved, else `None`.
        - 'anomalies_detected_count': The number of data points in the test set
          identified as anomalies by the algorithm.
        - 'train_set_size': The number of samples in the training set used.
        - 'test_set_size': The number of samples in the test/prediction set.

    Raises
    ------
    TypeError
        - If `feature_vector_data` (or resolved `train_data_var`/`test_data_var`)
          is not a pandas DataFrame or NumPy ndarray.
        - If `save_predictions_as` or `save_anomalous_indices_as` are provided but
          are not strings.
    ValueError
        - If `train_data_var` is provided without `test_data_var` or vice-versa.
        - If `train_split_ratio` is provided and is not between 0.0 and 1.0.
        - If `feature_vector_data` is too small to split (e.g., < 2 samples) when
          `train_split_ratio` is active.
        - If the predictions from LogAI are in an unexpected format (e.g., a multi-column
          DataFrame when a Series or 1D array is expected for scores/labels).
    Exception
        Any exception raised by LogAI's `AnomalyDetector` or the underlying
        scikit-learn (or other library) algorithm during model fitting or prediction.

    Examples
    --------
    >>> # Assume 'log_semantic_features' (a DataFrame from extract_log_features) is in session_vars.
    >>> detect_semantic_anomalies(
    ...     feature_vector_data="log_semantic_features",
    ...     algo_name="isolation_forest",
    ...     anomaly_detection_params={"n_estimators": 50, "contamination": 0.05},
    ...     train_split_ratio=0.8,
    ...     save_predictions_as="semantic_anomaly_preds",
    ...     save_anomalous_indices_as="semantic_anomalous_idxs"
    ... )
    >>> # Results summary is returned.
    >>> # session_vars["semantic_anomaly_preds"] contains prediction scores/labels.
    >>> # session_vars["semantic_anomalous_idxs"] contains the indices of anomalous logs.

    Notes
    -----
    - The interpretation of prediction values (e.g., what value signifies an anomaly)
      is specific to the `algo_name`. This tool makes a common assumption for
      'isolation_forest' (e.g., 1 means anomaly if LogAI wraps scikit-learn's output,
      or -1 if it's raw scikit-learn) and a generic one for others (e.g., < 0).
      This behavior might need to be adjusted based on thorough understanding of
      LogAI's wrappers for each algorithm.
    - If `feature_vector_data` is a NumPy array, it's converted to a pandas DataFrame
      internally to preserve original indices during splits.
    """
    import pandas as _pd
    import numpy as _np
    from sklearn.model_selection import train_test_split
    from logai.analysis.anomaly_detector import AnomalyDetectionConfig, AnomalyDetector
    
    _feature_vector = _resolve(feature_vector_data)
    if not isinstance(_feature_vector, (_pd.DataFrame, _np.ndarray)):
        raise TypeError(f"Expected 'feature_vector_data' to be DataFrame or ndarray, got {type(_feature_vector)}")

    if isinstance(_feature_vector, _np.ndarray):
        _feature_vector = _pd.DataFrame(_feature_vector)


    train_features: _pd.DataFrame
    test_features: _pd.DataFrame
    test_indices: _pd.Index 

    if train_data_var and test_data_var:
        train_features_resolved = _resolve(train_data_var)
        test_features_resolved = _resolve(test_data_var)
        if not isinstance(train_features_resolved, (_pd.DataFrame, _np.ndarray)) or \
           not isinstance(test_features_resolved, (_pd.DataFrame, _np.ndarray)):
            raise TypeError("Pre-split train/test data must be DataFrame or ndarray.")
        train_features = _pd.DataFrame(train_features_resolved)
        test_features = _pd.DataFrame(test_features_resolved)
        test_indices = test_features.index 
        logger.info(f"Using pre-split train data (var: {train_data_var}, shape: {train_features.shape}) and test data (var: {test_data_var}, shape: {test_features.shape}).")
    elif train_split_ratio is not None and (0 < train_split_ratio < 1):
        if len(_feature_vector) < 2 : 
             raise ValueError("Feature vector data is too small to split.")
        train_features, test_features = train_test_split(_feature_vector, train_size=train_split_ratio, shuffle=True) 
        test_indices = test_features.index
        logger.info(f"Splitting feature_vector_data with ratio {train_split_ratio}. Train shape: {train_features.shape}, Test shape: {test_features.shape}.")
    elif train_split_ratio is None: 
        train_features = _feature_vector
        test_features = _feature_vector
        test_indices = _feature_vector.index
        logger.info(f"No split ratio and no pre-split data. Training and predicting on full feature_vector_data (shape: {_feature_vector.shape}).")
    else:
        raise ValueError("Invalid combination of train_data_var, test_data_var, and train_split_ratio.")

    config = AnomalyDetectionConfig(
        algo_name=algo_name,
        algo_params=anomaly_detection_params or {}
    )
    anomaly_detector = AnomalyDetector(config)

    logger.info(f"Fitting {algo_name} model...")
    anomaly_detector.fit(train_features)

    logger.info(f"Predicting with {algo_name} model on test data...")
    predictions_raw = anomaly_detector.predict(test_features) 

    predictions_series: _pd.Series

    if isinstance(predictions_raw, _pd.DataFrame):
        if predictions_raw.shape[1] == 1:
            # If it's a single-column DataFrame, convert it to a Series
            predictions_series = predictions_raw.iloc[:, 0].copy()
            predictions_series.index = test_indices # Ensure index alignment
            logger.info("Raw predictions were a single-column DataFrame, converted to Series.")
        else:
            raise ValueError(
                f"Predictions from {algo_name} were a DataFrame with multiple columns ({predictions_raw.shape[1]}). "
                "Expected a Series, 1D numpy array, or single-column DataFrame."
            )
    elif isinstance(predictions_raw, _pd.Series):
        predictions_series = predictions_raw.copy()
        predictions_series.index = test_indices # Ensure index alignment
    elif isinstance(predictions_raw, _np.ndarray):
        if predictions_raw.ndim == 1:
            predictions_series = _pd.Series(predictions_raw, index=test_indices)
        elif predictions_raw.ndim == 2 and predictions_raw.shape[1] == 1:
            logger.info("Raw predictions were a 2D numpy array with one column, converting to Series.")
            predictions_series = _pd.Series(predictions_raw.flatten(), index=test_indices)
        else:
            raise ValueError(
                f"Predictions from {algo_name} were a numpy array with unexpected shape {predictions_raw.shape}. "
                "Expected 1D array or 2D array with one column."
            )
    else:
        try:
            # Last attempt to convert, though type is unknown
            predictions_series = _pd.Series(predictions_raw, index=test_indices)
            logger.warning(
                f"Predictions from {algo_name} were of an unexpected type ({type(predictions_raw)}). Attempted conversion to Series."
            )
        except Exception as e:
            raise TypeError(
                f"Predictions from {algo_name} (type: {type(predictions_raw)}) could not be reliably converted to pandas Series. Error: {e}"
            ) from e
    
    predictions_series.name = "anomaly_prediction"
        
    anomalies_count: int
    anomalous_data_indices: _pd.Index

    if algo_name == 'isolation_forest': 
        # Assuming LogAI's Isolation Forest wrapper might use 1 for anomaly
        # (standard sklearn uses -1 for outliers/anomalies, 1 for inliers)
        # This needs to be verified against LogAI's actual behavior for 'isolation_forest'
        boolean_mask = (predictions_series == 1) 
    else: 
        # Generic assumption: negative scores or scores != 1 (if 1 is normal) are anomalies.
        # This is a placeholder and ideally should be configured or made more robust based on algo behavior.
        # For demonstration, let's assume negative values are anomalous for non-IF cases.
        boolean_mask = (predictions_series < 0)
    
    anomalous_data_indices = predictions_series[boolean_mask].index
    anomalies_count = boolean_mask.sum() # sum() on a boolean Series gives count of True values

    results_summary = {
        "predictions_var": None,
        "anomalous_indices_var": None,
        "anomalies_detected_count": int(anomalies_count), # Ensure it's a Python int
        "train_set_size": len(train_features),
        "test_set_size": len(test_features)
    }

    if save_predictions_as:
        session_vars[save_predictions_as] = predictions_series
        results_summary["predictions_var"] = save_predictions_as
        logger.info(f"Saved raw prediction results to session variable: {save_predictions_as}")

    if save_anomalous_indices_as and not anomalous_data_indices.empty:
        session_vars[save_anomalous_indices_as] = anomalous_data_indices 
        results_summary["anomalous_indices_var"] = save_anomalous_indices_as
        logger.info(f"Saved {len(anomalous_data_indices)} anomalous indices to session variable: {save_anomalous_indices_as}")
    elif save_anomalous_indices_as:
        logger.info("No anomalies detected to save for anomalous_indices.")

    return results_summary

if __name__ == "__main__":
    app.run(transport="stdio")

