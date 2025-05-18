"""Preprocessing & parsing tools (clean → templates)."""

from typing import Any, List

import pandas as pd

from logai_mcp.session import (
    app,
    log_tool,
    _resolve,
    session_vars,
    logger,
)

# LogAI imports
from logai.preprocess.preprocessor import PreprocessorConfig, Preprocessor
from logai.utils import constants
from logai.information_extraction.log_parser import LogParser, LogParserConfig
from logai.algorithms.parsing_algo.drain import DrainParams
from typing import cast # Import cast

# --- Helper for robust list[str] conversion ---
def _ensure_list_of_strings(data: Any, logger_ref) -> List[str]:
    """Safely converts various data types to a list of strings."""
    res: List[str]
    if isinstance(data, list):
        res = [str(item) for item in data]
    elif isinstance(data, pd.Series):
        res = data.astype(str).tolist()
    elif isinstance(data, pd.DataFrame):
        logger_ref.warning(f"Attempting to convert DataFrame (shape: {data.shape}) to list of strings from its first column for Series fallback.")
        if data.empty:
            res = []
        else:
            res = data.iloc[:, 0].astype(str).tolist()
    elif hasattr(data, '__iter__') and not isinstance(data, (str, bytes, dict)): # Exclude single strings, bytes, dicts from iteration
        try:
            res = list(map(str, data))
        except Exception as e_map_str:
            logger_ref.error(f"Failed to map data (type: {type(data)}) to list of str: {e_map_str}. Returning empty list.")
            res = []
    # Scalar or unhandled non-iterable (including single strings that weren't caught as list-like)
    elif data is not None: # Changed from `if data is not None:` to `elif data is not None:`
        res = [str(data)] 
    else: # Handles data is None case
        res = []
    return cast(List[str], res) # Explicitly cast the return type

# ---------------------------------------------------------------------------
# Cleaning raw logs (regex replacements, etc.)
# ---------------------------------------------------------------------------


@app.tool()
@log_tool
def preprocess_log_data(
    log_record: Any,
    custom_replace_list: List[List[str]],
    *,
    save_clean_as: str, # For cleaned_logs (pd.Series)
    save_attributes_as: str, # For attributes (pd.DataFrame)
    save_patterns_as: str | None = None, # For custom_patterns (List[List[str]]), optional saving
    save_as: str | None = None,  # For the whole tuple result, optional saving
):
    """Clean raw log lines, extract attributes, and save key outputs (cleaned logs, attributes).

    Takes a `LogRecordObject` and applies regex replacements.
    - The cleaned log lines (Pandas Series) are **mandatorily** saved to `session_vars`
      under the key provided in `save_clean_as`.
    - The original attributes (Pandas DataFrame) are **mandatorily** saved to `session_vars`
      under the key provided in `save_attributes_as`.
    These named DataFrames/Series allow the LLM to directly use them for analysis.
    Optionally, the regex patterns used and the entire result tuple can also be saved.

    Parameters
    ----------
    log_record : Any
        The `LogRecordObject` containing the log data to be preprocessed.
        This argument is resolved from `session_vars` if it's a string key.
        Example: `"loaded_logs_var"` (a key in `session_vars`).
    custom_replace_list : List[List[str]]
        A list of regex replacement rules. Each rule is a list of two strings:
        `[pattern, replacement_string]`. The `pattern` is a regex to find,
        and `replacement_string` is what it's replaced with.
        Example: `[["\\b\\d{1,3}(\\.\\d{1,3}){3}\\b", "<IP>"]]` to replace IPs.
        This argument is **not** typically resolved from `session_vars`; it's usually
        provided directly or constructed by another tool/logic.
    save_clean_as : str
        **Required** key to store the `clean_logs` (pandas Series of cleaned log lines)
        in `session_vars`. Must be provided by the caller (LLM).
    save_attributes_as : str
        **Required** key to store the `attributes` (pandas DataFrame from the original
        `log_record`) in `session_vars`. Must be provided by the caller (LLM).
    save_patterns_as : str | None, default None
        Optional key to store the `custom_patterns` (the regex patterns applied)
        in `session_vars`. If `None`, patterns are not saved.
    save_as : str | None, default None
        Optional key to store the entire result tuple `(clean_logs, custom_patterns, attributes)`
        in `session_vars`. If `None`, the tuple is not saved under this combined key.

    Returns
    -------
    Tuple[pd.Series, List[List[str]], pd.DataFrame]
        A tuple containing:
        1. `clean_logs` (pd.Series): The log lines after applying regex replacements.
        2. `custom_patterns` (List[List[str]]): The list of regex patterns used.
        3. `attributes` (pd.DataFrame): The attributes extracted from the input `log_record`.

    Side Effects
    ------------
    - Stores `clean_logs` in `session_vars` (key: `save_clean_as` or auto-name).
    - Stores `attributes` in `session_vars` (key: `save_attributes_as` or auto-name).
    - Optionally stores `custom_patterns` in `session_vars` (key: `save_patterns_as`).
    - Optionally stores the entire result tuple in `session_vars` (key: `save_as`).

    Examples
    --------
    # Assuming `session_vars["raw_logs"]` is a LogRecordObject
    # and `replacements` is a list like [["pattern1", "REPL1"]]
    >>> preprocess_log_data(
    ...     log_record="raw_logs",
    ...     custom_replace_list=replacements,
    ...     save_clean_as="cleaned_log_lines",
    ...     save_attributes_as="original_attrs"
    ... )
    # session_vars["cleaned_log_lines"] will have cleaned logs.
    # session_vars["original_attrs"] will have the attributes.

    See Also
    --------
    load_file_log_data : Typically provides the `log_record` input.
    parse_log_data : Often the next step, taking `clean_logs` as input.
    _default_regex_replacements (in `pipelines.py`): For an example of `custom_replace_list`.

    Notes
    -----
    - The `log_record` input is resolved using `_resolve`.
    - The `custom_replace_list` is used by LogAI's `Preprocessor`.
    - `loglines` are extracted from `log_record.body[constants.LOGLINE_NAME]`.
    - `attributes` are from `log_record.attributes`.
    """
    log_record = _resolve(log_record)

    loglines = log_record.body[constants.LOGLINE_NAME]
    attributes = log_record.attributes

    preprocessor_config = PreprocessorConfig()
    preprocessor_config.custom_replace_list = custom_replace_list
    preprocessor = Preprocessor(preprocessor_config)

    clean_logs, custom_patterns = preprocessor.clean_log(loglines)

    # Mandatory saving for clean_logs and attributes
    session_vars[save_clean_as] = clean_logs
    logger.info(f"Saved cleaned logs (Series) to session_vars as '{save_clean_as}'.")

    session_vars[save_attributes_as] = attributes
    logger.info(f"Saved attributes (DataFrame) to session_vars as '{save_attributes_as}'.")

    # Optional saving for patterns
    if save_patterns_as:
        session_vars[save_patterns_as] = custom_patterns
        logger.info(f"Saved custom patterns to session_vars as '{save_patterns_as}'.")

    result = (clean_logs, custom_patterns, attributes)
    # Optional saving for the whole tuple
    if save_as:
        session_vars[save_as] = result
        logger.info(f"Saved entire preprocess_log_data tuple result to session_vars as '{save_as}'.")
    return result


# ---------------------------------------------------------------------------
# Parsing clean logs → templates (Drain)
# ---------------------------------------------------------------------------


@app.tool()
@log_tool
def parse_log_data(
    clean_logs: Any,
    parsing_algorithm: str = "drain",
    *,
    save_as: str,
):
    """Transform cleaned log lines into structured templates, returning a Pandas Series.

    Applies a log parsing algorithm (default "drain") to cleaned log lines
    (e.g., output from `preprocess_log_data`) to generate log templates.
    The resulting Pandas Series of templates is **mandatorily** stored in `session_vars`
    under the key provided in `save_as`. This allows the LLM to directly use
    the named Series of templates.

    Parameters
    ----------
    clean_logs : Any
        The cleaned log data to be parsed. This argument is resolved from
        `session_vars` if it's a string key. It can be:
        - A pandas Series of log strings.
        - A pandas DataFrame where the first column contains log strings.
        - A list or tuple of log strings.
        - A tuple where the first element is one of the above (e.g., output
          from `preprocess_log_data` before individual component saving).
        Example: `"cleaned_log_lines_var"` (a key in `session_vars`).
    parsing_algorithm : str, default "drain"
        The name of the parsing algorithm to use. LogAI supports several algorithms.
        "drain" is a common and robust choice.
        This argument is **not** resolved from `session_vars`.
    save_as : str
        The **required** key under which the Pandas Series of parsed log templates
        will be stored in `session_vars`. Must be provided by the caller (LLM).

    Returns
    -------
    pandas.Series
        A Series where each element is a parsed log template corresponding to an
        input log line. If parsing fails for some reason, it might fall back to
        returning the original `clean_logs` as a Series.

    Side Effects
    ------------
    - Stores the Series of parsed log templates in `session_vars` under the key
      specified by `save_as` or an auto-generated key.

    Examples
    --------
    # Assuming session_vars["cleaned_logs_output"] contains a Series of clean log lines:
    >>> parse_log_data(clean_logs="cleaned_logs_output", save_as="log_templates")
    # session_vars["log_templates"] will store the pandas Series of templates.

    # Using a different algorithm (if supported and configured in LogAI):
    >>> parse_log_data(clean_logs="cleaned_logs_output", parsing_algorithm="AEL", save_as="ael_templates")

    See Also
    --------
    preprocess_log_data : Usually provides the `clean_logs` input.
    vectorize_log_data : Often the next step, taking the parsed templates as input.

    Notes
    -----
    - The `clean_logs` input is resolved using `_resolve`.
    - Internally uses LogAI's `LogParser` with `DrainParams(sim_th=0.5, depth=5)`
      by default when `parsing_algorithm` is "drain".
    - If `clean_logs` is a tuple (e.g., direct output from `preprocess_log_data`
      before individual saving), the tool extracts the first element.
    - Input is coerced into a `list[str]` before parsing.
    - If parsing raises an exception, a warning is logged, and the function attempts
      to return the original `clean_logs` as a `pd.Series`.
    - The output structure from the parser (`parsed_result["parsed_logline"]`)
      is extracted to get the templates.
    """
    clean_logs = _resolve(clean_logs)

    if isinstance(clean_logs, tuple) and len(clean_logs) >= 1:
        clean_logs = clean_logs[0]

    # Convert Series / DataFrame to list[str]
    if isinstance(clean_logs, pd.Series):
        clean_logs = clean_logs.astype(str).tolist()
    elif isinstance(clean_logs, pd.DataFrame):
        clean_logs = clean_logs.iloc[:, 0].astype(str).tolist()

    params = DrainParams(sim_th=0.5, depth=5)
    cfg = LogParserConfig()
    cfg.parsing_algorithm = parsing_algorithm
    cfg.parsing_algo_params = params
    parser = LogParser(cfg)

    try:
        parsed_result = parser.parse(clean_logs)  # type: ignore[arg-type]
    except Exception as exc:
        logger.warning(
            "Falling back to raw lines because parsing failed: %s: %s", type(exc).__name__, exc
        )
        # Ensure clean_logs is a list before creating a Series in this fallback
        if not isinstance(clean_logs, list):
            # This case should ideally be rare if prior conversions worked,
            # but as a safeguard:
            try:
                # Attempt to convert to list of strings if possible
                clean_logs = list(map(str, clean_logs))
            except TypeError:
                # If not iterable or convertible, fallback to an empty list
                logger.error(f"Cannot convert clean_logs of type {type(clean_logs)} to list for Series fallback. Using empty list.")
                clean_logs = []
        parsed_result = pd.Series(clean_logs)

    try:
        parsed_loglines = parsed_result["parsed_logline"]
    except (TypeError, KeyError):
        # If parsed_result is already the series (e.g. fallback)
        parsed_loglines = parsed_result # This might be pd.Series(clean_logs)

    # Ensure parsed_loglines is a pd.Series before saving
    if not isinstance(parsed_loglines, pd.Series):
        try:
            # Attempt to convert to Series if it's list-like, e.g. from parser output
            # Ensure the input is a list of strings before creating the Series
            list_of_strings_for_series = _ensure_list_of_strings(parsed_loglines, logger)
            parsed_loglines = pd.Series(list_of_strings_for_series, dtype=str)
        except Exception as e_conv: # Broad exception if conversion fails
            logger.warning(f"Could not convert parsed_loglines (type: {type(parsed_loglines)}) to pd.Series directly: {e_conv}. Attempting fallback using original clean_logs.")
            # Fallback to original clean_logs as series if all else fails.
            
            list_from_helper: List[str] = _ensure_list_of_strings(clean_logs, logger)
            
            # Explicitly check if the helper returned what we expect (List[str])
            if isinstance(list_from_helper, list):
                 # Further ensure all items are strings, though _ensure_list_of_strings should handle this.
                # This is an extra guard for Pylance.
                list_from_helper_all_str = [str(item) for item in list_from_helper]
                parsed_loglines = pd.Series(list_from_helper_all_str, dtype=str) # type: ignore
            else:
                # This case should ideally not be reached if _ensure_list_of_strings works as intended.
                logger.error(f"Fallback helper _ensure_list_of_strings did not return a List as expected. "
                             f"Received type: {type(list_from_helper)}. Defaulting to empty Series.")
                parsed_loglines = pd.Series([], dtype=str) # Default to empty Series of strings

    session_vars[save_as] = parsed_loglines
    logger.info(f"Saved parsed log templates (Series) to session_vars as '{save_as}'.")
    return parsed_loglines
