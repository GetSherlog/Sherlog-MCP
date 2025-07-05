from IPython.core.interactiveshell import ExecutionResult

from sherlog_mcp.tools.tool_utils import error_dict, success_dict


def return_result(code: str, execution_result: ExecutionResult, command: str, save_as: str) -> dict:
    if execution_result and hasattr(execution_result, 'result') and execution_result.result is not None:
        message = f"Command executed and saved to '{save_as}'"
        response = success_dict(message, command=command, saved_as=save_as)
        return response
    elif execution_result and execution_result.error_in_exec is not None:
        return error_dict(f"Encountered error during execution: {command}", str(execution_result.error_in_exec))
    elif execution_result and execution_result.error_before_exec is not None:
        return error_dict(f"Encountered error before code execution: {code}", str(execution_result.error_before_exec))
    return error_dict(f"Failed to execute command: {command}", "No result returned from shell execution")