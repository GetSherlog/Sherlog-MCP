import docker
import pandas as pd

from sherlog_mcp.dataframe_utils import smart_create_dataframe, to_pandas
from sherlog_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from sherlog_mcp.session import (
    app,
    logger,
)


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
    logger.info("Listing Docker containers...")
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
    logger.info(f"Found {len(container_list)} containers.")
    # Use smart DataFrame creation for better performance with polars when available
    container_data_df = smart_create_dataframe(container_list, prefer_polars=True)
    # Convert to pandas for compatibility with existing code
    return to_pandas(container_data_df)


_SHELL.push({"list_containers_impl": _list_containers_impl})


# Conditional tool registration based on Docker availability
if _docker_available():
    logger.info("Docker daemon detected - registering Docker tools")

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
                          (ID, Name, Image, Status). Returns an empty DataFrame if
                          no containers are found or if an error occurs (with a logged error).

        """
        code = f"{save_as} = list_containers_impl()\n" + f"{save_as}"
        execution_result = await run_code_in_shell(code)
        df = execution_result.result if execution_result else None
        if isinstance(df, pd.DataFrame):
            return df.to_dict("records")

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

        """
        code = (
            f'{save_as} = get_container_logs_impl("{container_id}", "{tail}")\n'
            + f"{save_as}"
        )
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

else:
    logger.info("Docker daemon not detected - Docker tools will not be registered")
