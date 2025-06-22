"""Preprocessing & parsing tools (clean → templates)."""

from typing import Any, cast

import pandas as pd
from logai.algorithms.parsing_algo.drain import DrainParams
from logai.dataloader.data_loader import LogRecordObject
from logai.information_extraction.log_parser import LogParser, LogParserConfig
from logai.preprocess.preprocessor import Preprocessor, PreprocessorConfig
from logai.utils import constants

from sherlog_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from sherlog_mcp.session import (
    app,
    logger,
)


def _ensure_list_of_strings(data: Any, logger_ref) -> list[str]:
    """Safely converts various data types to a list of strings."""
    res: list[str]
    if isinstance(data, list):
        res = [str(item) for item in data]
    elif isinstance(data, pd.Series):
        res = data.astype(str).tolist()
    elif isinstance(data, pd.DataFrame):
        logger_ref.warning(
            f"Attempting to convert DataFrame (shape: {data.shape}) to list of strings from its first column for Series fallback."
        )
        if data.empty:
            res = []
        else:
            res = data.iloc[:, 0].astype(str).tolist()
    elif hasattr(data, "__iter__") and not isinstance(
        data, (str, bytes, dict)
    ):
        try:
            res = list(map(str, data))
        except Exception as e_map_str:
            logger_ref.error(
                f"Failed to map data (type: {type(data)}) to list of str: {e_map_str}. Returning empty list."
            )
            res = []
    elif data is not None:
        res = [str(data)]
    else:
        res = []
    return cast(list[str], res)


def _preprocess_log_data_impl(
    log_record: LogRecordObject,
    custom_replace_list: list[list[str]],
) -> tuple[pd.Series, list[list[str]], pd.DataFrame]:
    """Clean raw log lines, extract attributes. Returns (cleaned_logs, patterns, attributes).
    
    The outputs persist as save_clean_as, save_patterns_as, save_attributes_as.
    Use execute_python_code() to work with cleaned data, e.g. "{save_clean_as}.head()"
    Data parameters can be DataFrame variables from previous tool calls.
    Use list_dataframes() to see available DataFrames."""
    loglines = log_record.body[constants.LOGLINE_NAME]  # type: ignore
    attributes = log_record.attributes  # type: ignore

    preprocessor_config = PreprocessorConfig()
    preprocessor_config.custom_replace_list = custom_replace_list
    preprocessor = Preprocessor(preprocessor_config)

    clean_logs, custom_patterns = preprocessor.clean_log(loglines)

    return clean_logs, custom_patterns, attributes


_SHELL.push({"_preprocess_log_data_impl": _preprocess_log_data_impl})


@app.tool()
async def preprocess_log_data(
    log_record: str | LogRecordObject,
    custom_replace_list: list[list[str]],
    *,
    save_clean_as: str,
    save_attributes_as: str,
    save_patterns_as: str | None = None,
):
    """Wrapper for `_preprocess_log_data_impl`, saves outputs to shell variables."""
    code = f"{save_clean_as}, {save_patterns_as}, {save_attributes_as} = _preprocess_log_data_impl({log_record}, {repr(custom_replace_list)})\n"
    execution_result = await run_code_in_shell(code)
    return execution_result.result if execution_result else None


preprocess_log_data.__doc__ = _preprocess_log_data_impl.__doc__


def _parse_log_data_impl(
    clean_logs: pd.Series | pd.DataFrame | list[str],
    parsing_algorithm: str = "drain",
) -> pd.Series:
    """Transform cleaned log lines into structured templates, returning a Pandas Series."""
    if isinstance(clean_logs, pd.Series):
        logs_to_parse = clean_logs.astype(str).tolist()
    elif isinstance(clean_logs, pd.DataFrame):
        logs_to_parse = clean_logs.iloc[:, 0].astype(str).tolist()
    elif isinstance(clean_logs, list):
        logs_to_parse = [str(x) for x in clean_logs]
    else:
        raise TypeError(
            f"_parse_log_data_impl expects clean_logs to be Series, DataFrame, or list, got {type(clean_logs)}"
        )

    params = DrainParams()
    params.sim_th = 0.5  # type: ignore[attr-defined]
    params.depth = 5  # type: ignore[attr-defined]

    cfg = LogParserConfig()
    cfg.parsing_algorithm = parsing_algorithm
    cfg.parsing_algo_params = params
    parser = LogParser(cfg)

    try:
        parsed_result_dict = parser.parse(pd.Series(logs_to_parse, dtype=str))
        parsed_loglines_series = parsed_result_dict["parsed_logline"]
    except Exception as exc:
        logger.warning(
            "Parsing failed: %s: %s. Falling back to original clean_logs.",
            type(exc).__name__,
            exc,
        )
        parsed_loglines_series = pd.Series(logs_to_parse, dtype=str)

    if not isinstance(parsed_loglines_series, pd.Series):
        logger.error(
            f"Parsed result was not a Series (type: {type(parsed_loglines_series)}). Defaulting to Series from input logs."
        )
        parsed_loglines_series = pd.Series(
            _ensure_list_of_strings(logs_to_parse, logger), dtype=str
        )

    return parsed_loglines_series


_SHELL.push({"_parse_log_data_impl": _parse_log_data_impl})


@app.tool()
async def parse_log_data(
    clean_logs: str | pd.Series | pd.DataFrame | list[str],
    parsing_algorithm: str = "drain",
    *,
    save_as: str,
):
    """Wrapper for `_parse_log_data_impl`, saves result to shell variable."""
    code = (
        f"{save_as} = _parse_log_data_impl({clean_logs}, {repr(parsing_algorithm)})\n"
        f"{save_as}"
    )
    execution_result = await run_code_in_shell(code)
    return execution_result.result if execution_result else None


parse_log_data.__doc__ = _parse_log_data_impl.__doc__
