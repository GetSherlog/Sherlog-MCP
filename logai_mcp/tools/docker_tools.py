from typing import Union # Add Union import
import docker
from docker.errors import NotFound, DockerException
import pandas as pd
import json

from logai_mcp.session import (
    app,
    log_tool,
    _resolve,
    session_vars,
    logger,
)

@app.tool()
@log_tool
def list_containers(*, save_as: str):
    """
    Lists all running Docker containers and returns them as a Pandas DataFrame.

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
    container_data_df = pd.DataFrame() # Default to empty DataFrame
    try:
        if session_vars: # session_vars is globally available via import
            client = session_vars.get('docker_client')
            if client is None: # Ensure client is initialized
                client = docker.from_env()
                session_vars['docker_client'] = client # Store if newly created
        else: # This case should ideally not happen if session_vars is always available
            client = docker.from_env()
            # Consider if session_vars should be initialized here if it's None,
            # but typically it's managed by the session module.

        containers = client.containers.list()
        if containers:
            container_list = []
            for container in containers:
                container_list.append({
                    "ID": container.short_id,
                    "Name": container.name,
                    "Image": container.attrs['Config']['Image'],
                    "Status": container.status
                })
            container_data_df = pd.DataFrame(container_list)
        else:
            logger.info("No running Docker containers found.")
            # container_data_df remains an empty DataFrame
            
    except DockerException as e:
        logger.error(f"Error listing Docker containers: {str(e)}")
        # container_data_df remains an empty DataFrame, error is logged
    except Exception as e:
        logger.error(f"An unexpected error occurred while listing containers: {str(e)}")
        # container_data_df remains an empty DataFrame, error is logged

    # Auto-naming logic was here, removed as save_as is now mandatory.
    session_vars[save_as] = container_data_df # Save the DataFrame
    logger.info(f"Saved listed containers (DataFrame) to session_vars as '{save_as}'.")
    return container_data_df # Return the DataFrame

@app.tool()
@log_tool
def get_container_logs(container_id: str, tail: Union[str, int] = "all", *, save_as: str):
    """
    Retrieves logs for a specific Docker container and saves them as a string.

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
    result = None
    try:
        client = session_vars.get('docker_client')
        if client is None:
            client = docker.from_env()
        session_vars['docker_client'] = client

        container = client.containers.get(container_id)
        logs = container.logs(tail=tail if tail == "all" else int(tail), timestamps=True)
        result = logs.decode('utf-8')
    except NotFound:
        result = f"Error: Container '{container_id}' not found."
    except DockerException as e:
        result = f"Error retrieving logs for container '{container_id}': {str(e)}"
    except ValueError:
        result = f"Error: Invalid value for 'tail'. Must be 'all' or an integer."
    except Exception as e:
        result = f"An unexpected error occurred: {str(e)}"

    # Auto-naming logic was here, removed as save_as is now mandatory.
    session_vars[save_as] = result
    logger.info(f"Saved container logs/error to session_vars as '{save_as}'.")
    return result
