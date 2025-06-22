import docker
import pandas as pd

from sherlog_mcp.dataframe_utils import smart_create_dataframe, to_pandas
from sherlog_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from sherlog_mcp.session import (
    app,
    logger,
)
from sherlog_mcp.tools.preprocessing import _parse_log_data_impl


def _docker_available() -> bool:
    """Check if Docker daemon is accessible."""
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def _list_containers_impl() -> pd.DataFrame | None:
    """Lists all running Docker containers and returns them as a Pandas DataFrame.

    The resulting Pandas DataFrame, containing details of running containers
    (ID, Name, Image, Status), is **mandatorily** stored in `session_vars`
    under the key provided in `save_as`. This allows the LLM to directly use
    this named DataFrame.

    Args:
        save_as (str): The **required** key under which the listed
                       containers (Pandas DataFrame) will be stored
                       in `session_vars`. Must be provided by the caller (LLM).

    Returns:
        pd.DataFrame: A Pandas DataFrame containing details of running containers
                      (ID, Name, Image, Status). Returns an empty DataFrame if
                      no containers are found or if an error occurs (with a logged error).

    """
    client = docker.from_env()
    containers = client.containers.list()
    if not containers:
        return None
    container_list = []
    for container in containers:
        container_list.append(
            {
                "ID": container.short_id,
                "Name": container.name,
                "Image": container.attrs["Config"]["Image"],
                "Status": container.status,
            }
        )
    container_data_df = smart_create_dataframe(container_list, prefer_polars=True)
    return to_pandas(container_data_df)


_SHELL.push({"list_containers_impl": _list_containers_impl})


if _docker_available():

    @app.tool()
    async def list_containers(*, save_as: str) -> pd.DataFrame | None:
        """Lists all running Docker containers and returns them as a Pandas DataFrame.

        The resulting Pandas DataFrame, containing details of running containers
        (ID, Name, Image, Status), is **mandatorily** stored in `session_vars`
        under the key provided in `save_as`. This allows the LLM to directly use
        this named DataFrame.

        Args:
            save_as (str): The **required** key under which the listed
                           containers (Pandas DataFrame) will be stored
                           in `session_vars`. Must be provided by the caller (LLM).

        Returns:
            pd.DataFrame: A Pandas DataFrame containing details of running containers
                          (ID, Name, Image, Status). Returns `None` if no containers
                          are found or if an error occurs (with a logged error).

        Examples
        --------
        After calling this tool with save_as="containers":
        
        # View all containers
        >>> execute_python_code("containers")
        
        # View first 5 containers
        >>> execute_python_code("containers.head()")
        
        # Check how many containers are running
        >>> execute_python_code("len(containers)")
        >>> execute_python_code("containers.shape")
        
        # Filter containers by image
        >>> execute_python_code("containers[containers['Image'].str.contains('nginx')]")
        
        # View specific columns
        >>> execute_python_code("containers[['Name', 'Status', 'Image']]")
        
        # Get container IDs as a list
        >>> execute_python_code("containers['ID'].tolist()")

        """
        code = f"{save_as} = list_containers_impl()\n" + f"{save_as}"
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    def _get_container_logs_impl(container_id: str, tail: str | int = "all") -> str:
        client = docker.from_env()
        container = client.containers.get(container_id)
        logs = container.logs(
            tail=tail if tail == "all" else int(tail), timestamps=True
        )
        result = logs.decode("utf-8")
        return result

    _SHELL.push({"get_container_logs_impl": _get_container_logs_impl})

    @app.tool()
    async def get_container_logs(
        container_id: str, tail: str | int = "all", *, save_as: str
    ) -> str | None:
        """Retrieves logs for a specific Docker container and saves them as a string.

        The container logs (or an error message) are returned as a string and
        **mandatorily** stored in `session_vars` under the key provided in `save_as`.
        The LLM must provide this name. While logs are strings, not DataFrames,
        consistent saving behavior is applied.

        Args:
            container_id (str): The ID or name of the container.
            tail (Union[str, int]): Number of lines to show from the end of the logs.
                                    "all" to show all logs. Defaults to "all".
            save_as (str): The **required** key under which the container
                           logs (string) or error message (string) will be stored
                           in `session_vars`. Must be provided by the caller (LLM).

        Returns:
            str: The container logs as a string, or an error message string.
            
        Examples
        --------
        After calling this tool with save_as="logs":
        
        # View first 1000 characters
        >>> execute_python_code("print(logs[:1000])")
        
        # View last 500 characters
        >>> execute_python_code("print(logs[-500:])")
        
        # Count total lines
        >>> execute_python_code("len(logs.splitlines())")
        
        # Search for errors
        >>> execute_python_code("[line for line in logs.splitlines() if 'error' in line.lower()][:10]")
        
        # View logs with line numbers
        >>> execute_python_code("for i, line in enumerate(logs.splitlines()[:20]): print(f'{i+1}: {line}')")

        """
        code = (
            f'{save_as} = get_container_logs_impl("{container_id}", "{tail}")\n'
            + f"{save_as}"
        )
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    def _get_container_logs_structured_impl(
        container_id: str, tail: str | int = "all"
    ) -> pd.DataFrame | None:
        """Fetch Docker logs and parse them into structured templates using DRain.

        The resulting DataFrame has at minimum the columns:

        • ``timestamp`` – as ISO-8601 string (extracted from each log line if present)
        • ``message`` – raw log message text after the timestamp
        • ``template`` – DRain log template produced by :pyfunc:`_parse_log_data_impl`
        """
        raw_text = _get_container_logs_impl(container_id, tail)
        if raw_text is None:
            return None  # pragma: no cover

        rows: list[dict[str, str]] = []
        messages: list[str] = []

        for line in raw_text.splitlines():
            if not line.strip():
                continue
            if " " in line:
                ts, msg = line.split(" ", 1)
            else:
                ts, msg = "", line
            rows.append({"timestamp": ts, "message": msg})
            messages.append(msg)

        if not rows:
            return pd.DataFrame()

        try:
            templates_series = _parse_log_data_impl(messages, parsing_algorithm="drain")
        except Exception as exc:
            templates_series = pd.Series([None] * len(messages), name="template")

        df = pd.DataFrame(rows)
        df["template"] = templates_series.values
        return to_pandas(df)

    _SHELL.push({"_get_container_logs_structured_impl": _get_container_logs_structured_impl})

    @app.tool()
    async def get_container_logs_structured(
        container_id: str,
        tail: str | int = "all",
        *,
        save_as: str,
    ) -> pd.DataFrame | None:
        """Retrieve Docker logs and return a Drain-parsed structured DataFrame.

        This is a convenience wrapper that chains :pyfunc:`get_container_logs` with
        LogAI's Drain parser so the caller gets structured output in one step.

        Args:
            container_id: The ID or name of the container.
            tail: Number of lines from the end ("all" for entire logs).
            save_as: Variable name to store the resulting DataFrame in the IPython shell.

        Returns
        -------
        pandas.DataFrame | None
            DataFrame with columns ``timestamp``, ``message``, ``template``.  ``None``
            if log retrieval fails or no logs are available.
            
        Examples
        --------
        After calling this tool with save_as="structured_logs":
        
        # View all structured logs
        >>> execute_python_code("structured_logs")
        
        # View first 10 log entries
        >>> execute_python_code("structured_logs.head(10)")
        
        # Group by template to find patterns
        >>> execute_python_code("structured_logs['template'].value_counts().head(10)")
        
        # Filter for specific templates or patterns
        >>> execute_python_code("structured_logs[structured_logs['message'].str.contains('error', case=False)]")
        
        # View logs from a specific time range
        >>> execute_python_code("structured_logs[structured_logs['timestamp'] > '2025-01-01T00:00:00']")
        
        # Analyze message lengths
        >>> execute_python_code("structured_logs['message'].str.len().describe()")
        
        # Export to CSV for external analysis
        >>> execute_python_code("structured_logs.to_csv('container_logs.csv', index=False)")
        """
        code = (
            f"{save_as} = _get_container_logs_structured_impl(\"{container_id}\", \"{tail}\")\n"
            f"{save_as}"
        )
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

else:
    logger.info("Docker daemon not detected - Docker tools will not be registered")
