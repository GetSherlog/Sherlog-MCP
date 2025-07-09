from IPython.core.interactiveshell import ExecutionResult
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def dataframe_to_dict(df: Any, saved_as: str, message: str = "Operation completed") -> dict:
    """
    Convert DataFrame result to standardized dict response.
    
    Args:
        df: DataFrame or result from shell execution
        saved_as: Variable name where data is saved in IPython namespace
        message: Success message
        
    Returns:
        Standardized dict with DataFrame metadata
    """
    if df is None:
        return {
            "success": False,
            "message": "No data returned",
            "error": "Result was None"
        }
    
    response = {
        "success": True,
        "message": message,
        "saved_as": saved_as,
        "note": f"Full results saved as '{saved_as}' in IPython namespace"
    }
    
    if hasattr(df, 'shape'):
        response["rows"] = df.shape[0]
        response["columns"] = list(df.columns) if hasattr(df, 'columns') else []
        response["sample"] = df.head(5).to_dict('records') if hasattr(df, 'to_dict') else []
    elif hasattr(df, '__len__'):
        response["count"] = len(df)
        if isinstance(df, list):
            response["sample"] = df[:5] if len(df) > 5 else df
    
    return response


def shell_result_to_dict(result: Any, default_message: str = "Operation failed") -> dict:
    """
    Process shell execution result and return standardized dict.
    
    Args:
        result: Result from run_code_in_shell
        default_message: Default error message
        
    Returns:
        Either the DataFrame dict or error dict
    """
    if result and hasattr(result, 'result') and result.result is not None:
        return result.result if isinstance(result.result, dict) else {"data": result.result}
    
    return {
        "success": False,
        "message": default_message,
        "error": "No result returned from shell execution"
    }


def error_dict(message: str, error: Optional[str] = None) -> dict:
    """
    Create a standardized error response dict.
    
    Args:
        message: Error message
        error: Optional detailed error information
        
    Returns:
        Standardized error dict
    """
    response = {
        "success": False,
        "message": message
    }
    if error:
        response["error"] = error
    return response


def success_dict(message: str, **kwargs) -> dict:
    """
    Create a standardized success response dict.
    
    Args:
        message: Success message
        **kwargs: Additional fields to include in response
        
    Returns:
        Standardized success dict
    """
    response = {
        "success": True,
        "message": message
    }
    response.update(kwargs)
    return response


def return_result(code: str, execution_result: ExecutionResult, command: str, save_as: str) -> dict:
    if execution_result and hasattr(execution_result, 'result') and execution_result.result is not None:
        message = f"Command executed and saved to '{save_as}'"
        response = success_dict(message, command=command, saved_as=save_as)
        return response
    elif execution_result and execution_result.error_in_exec is not None:
        return error_dict(f"Encountered error during execution: {command}", str(execution_result.error_in_exec))
    elif execution_result and execution_result.error_before_exec is not None:
        return error_dict(f"Encountered error before code execution: {code}", str(execution_result.error_before_exec))
    
    # The shell has silent=True so we dont see any execution result
    if execution_result.success:
        message = f"Command executed and saved to '{save_as}'"
        response = success_dict(message, command=command, saved_as=save_as)
        return response
    else:
        return error_dict(f"Failed to execute command: {command}", "No result returned from shell execution")