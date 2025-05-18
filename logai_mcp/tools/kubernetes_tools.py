"""
Tools for interacting with a Kubernetes MCP server using the mcp Python SDK.

These tools act as clients to an external Kubernetes MCP server
(e.g., https://github.com/manusa/kubernetes-mcp-server),
fetch data, and return it as Pandas DataFrames.
The communication is handled via stdio using the mcp Python SDK.
"""
import pandas as pd
from typing import Optional, List, Dict, Any, Union
import asyncio # Required for async operations

# MCP Python SDK imports
from mcp import ClientSession, StdioServerParameters, types as mcp_types
from mcp.client.stdio import stdio_client

# Imports from logai_mcp.session, similar to other tool files
from logai_mcp.session import (
    app,
    log_tool,
    session_vars,
    logger,
)

# Configuration for the Kubernetes MCP server
KUBERNETES_MCP_COMMAND = "npx"
KUBERNETES_MCP_ARGS = ["-y", "kubernetes-mcp-server@latest"]

# Define StdioServerParameters for the Kubernetes MCP server
kubernetes_server_params = StdioServerParameters(
    command=KUBERNETES_MCP_COMMAND,
    args=KUBERNETES_MCP_ARGS,
    env=None, # Add environment variables if needed
)

async def _call_kubernetes_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
    """
    Internal helper to call an external Kubernetes MCP server tool via stdio.
    Returns the raw response from the server or None if an error occurs.
    """
    # This print is for low-level debugging; formal logging is done by the calling public tool.
    print(f"Attempting Kubernetes MCP call: {tool_name} with args: {arguments} via stdio")
    try:
        async with stdio_client(kubernetes_server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                response = await session.call_tool(tool_name, arguments=arguments)
        return response
    except Exception as e:
        # This print is for low-level debugging of SDK call issues.
        print(f"Error during Kubernetes MCP SDK call {tool_name} with args {arguments}: {e}")
        # The public tool function will log this error more formally using the logger.
        return None

@app.tool()
@log_tool
async def view_kubernetes_configuration(minified: Optional[bool] = True, *, save_as: str) -> pd.DataFrame:
    """
    Retrieves the current Kubernetes configuration (kubeconfig) as a YAML string.

    This tool wraps the 'configuration_view' tool from the external Kubernetes MCP server.
    It fetches the kubeconfig, which can be either the full version or a minified one
    containing only the current context and relevant details. The resulting YAML string
    is placed into a single-cell Pandas DataFrame.

    The output DataFrame is **mandatorily** saved to `session_vars` under the key
    specified by `save_as`.

    Parameters
    ----------
    minified : Optional[bool], default True
        If True, returns a minified version of the configuration (current context only).
        If False, returns the full configuration.
    save_as : str
        The **required** key under which the resulting Pandas DataFrame (containing
        the kubeconfig YAML string) will be stored in `session_vars`.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with a single column 'configuration' and a single row
        containing the kubeconfig YAML string. Returns an empty DataFrame if an
        error occurs or the configuration cannot be retrieved.

    Side Effects
    ------------
    - Stores the output DataFrame in `session_vars` (key: `save_as`).
    - Logs information about the operation success or failure.
    """
    args = {"minified": minified}
    tool_name = "configuration_view"
    df_result = pd.DataFrame()
    try:
        raw_response = await _call_kubernetes_mcp_tool(tool_name, args)
        if raw_response and isinstance(raw_response, dict) and "config" in raw_response:
            df_result = pd.DataFrame([{"configuration": raw_response["config"]}])
            logger.info(f"Successfully retrieved Kubernetes configuration. Saved to '{save_as}'.")
        elif raw_response is None: 
            logger.error(f"Failed to retrieve Kubernetes configuration for tool '{tool_name}' with args {args}. SDK call failed.")
        else:
            logger.error(f"Unexpected response structure for tool '{tool_name}' with args {args}: {raw_response}")
    except Exception as e:
        logger.error(f"Exception in view_kubernetes_configuration with args {args}: {e}")
    
    session_vars[save_as] = df_result
    return df_result

@app.tool()
@log_tool
async def list_kubernetes_events(namespace: Optional[str] = None, *, save_as: str) -> pd.DataFrame:
    """
    Lists Kubernetes events from the cluster.

    This tool wraps the 'events_list' tool from the external Kubernetes MCP server.
    It can list events from a specific namespace or from all namespaces if none is specified.
    The events are returned as a Pandas DataFrame.

    The output DataFrame is **mandatorily** saved to `session_vars` under the key
    specified by `save_as`.

    Parameters
    ----------
    namespace : Optional[str], default None
        The Kubernetes namespace from which to list events. If None, events from all
        namespaces are listed.
    save_as : str
        The **required** key under which the resulting Pandas DataFrame (containing
        event details) will be stored in `session_vars`.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame where each row represents a Kubernetes event, with columns
        corresponding to event attributes (e.g., namespace, lastTimestamp, reason, message).
        Returns an empty DataFrame if an error occurs or no events are found.

    Side Effects
    ------------
    - Stores the output DataFrame in `session_vars` (key: `save_as`).
    - Logs information about the operation success or failure.
    """
    args = {}
    if namespace:
        args["namespace"] = namespace
    tool_name = "events_list"
    df_result = pd.DataFrame()
    try:
        raw_response = await _call_kubernetes_mcp_tool(tool_name, args)
        if raw_response and isinstance(raw_response, dict) and "events" in raw_response:
            df_result = pd.DataFrame(raw_response["events"])
            logger.info(f"Successfully listed Kubernetes events. Saved to '{save_as}'.")
        elif raw_response is None:
            logger.error(f"Failed to list Kubernetes events for tool '{tool_name}' with args {args}. SDK call failed.")
        else:
            logger.error(f"Unexpected response structure for tool '{tool_name}' with args {args}: {raw_response}")
    except Exception as e:
        logger.error(f"Exception in list_kubernetes_events with args {args}: {e}")

    session_vars[save_as] = df_result
    return df_result

@app.tool()
@log_tool
async def list_helm_releases(namespace: Optional[str] = None, all_namespaces: Optional[bool] = False, *, save_as: str) -> pd.DataFrame:
    """
    Lists Helm releases in the Kubernetes cluster.

    This tool wraps the 'helm_list' tool from the external Kubernetes MCP server.
    It can list Helm releases from a specific namespace, all namespaces (if `all_namespaces`
    is True), or the configured default namespace. The releases are returned as a Pandas DataFrame.

    The output DataFrame is **mandatorily** saved to `session_vars` under the key
    specified by `save_as`.

    Parameters
    ----------
    namespace : Optional[str], default None
        The namespace from which to list Helm releases. If not provided, uses the
        configured default namespace (behavior of the underlying MCP tool).
    all_namespaces : Optional[bool], default False
        If True, lists Helm releases from all namespaces. Overrides `namespace` if both are set.
    save_as : str
        The **required** key under which the resulting Pandas DataFrame (containing
        Helm release details) will be stored in `session_vars`.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame where each row represents a Helm release, with columns
        for attributes like name, namespace, revision, and status.
        Returns an empty DataFrame if an error occurs or no releases are found.

    Side Effects
    ------------
    - Stores the output DataFrame in `session_vars` (key: `save_as`).
    - Logs information about the operation success or failure.
    """
    args = {}
    if namespace:
        args["namespace"] = namespace
    if all_namespaces:
        args["all_namespaces"] = True # MCP tool expects boolean
    tool_name = "helm_list"
    df_result = pd.DataFrame()
    try:
        raw_response = await _call_kubernetes_mcp_tool(tool_name, args)
        if raw_response and isinstance(raw_response, dict) and "releases" in raw_response:
            df_result = pd.DataFrame(raw_response["releases"])
            logger.info(f"Successfully listed Helm releases. Saved to '{save_as}'.")
        elif raw_response is None:
            logger.error(f"Failed to list Helm releases for tool '{tool_name}' with args {args}. SDK call failed.")
        else:
            logger.error(f"Unexpected response structure for tool '{tool_name}' with args {args}: {raw_response}")
    except Exception as e:
        logger.error(f"Exception in list_helm_releases with args {args}: {e}")

    session_vars[save_as] = df_result
    return df_result

@app.tool()
@log_tool
async def list_kubernetes_namespaces(*, save_as: str) -> pd.DataFrame:
    """
    Lists all Kubernetes namespaces in the current cluster.

    This tool wraps the 'namespaces_list' tool from the external Kubernetes MCP server.
    The namespaces are returned as a Pandas DataFrame.

    The output DataFrame is **mandatorily** saved to `session_vars` under the key
    specified by `save_as`.

    Parameters
    ----------
    save_as : str
        The **required** key under which the resulting Pandas DataFrame (containing
        namespace details like name and status) will be stored in `session_vars`.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame where each row represents a Kubernetes namespace.
        Returns an empty DataFrame if an error occurs or no namespaces are found.

    Side Effects
    ------------
    - Stores the output DataFrame in `session_vars` (key: `save_as`).
    - Logs information about the operation success or failure.
    """
    args = {}
    tool_name = "namespaces_list"
    df_result = pd.DataFrame()
    try:
        raw_response = await _call_kubernetes_mcp_tool(tool_name, args)
        if raw_response and isinstance(raw_response, dict) and "namespaces" in raw_response:
            df_result = pd.DataFrame(raw_response["namespaces"])
            logger.info(f"Successfully listed Kubernetes namespaces. Saved to '{save_as}'.")
        elif raw_response is None:
            logger.error(f"Failed to list Kubernetes namespaces for tool '{tool_name}' with args {args}. SDK call failed.")
        else:
            logger.error(f"Unexpected response structure for tool '{tool_name}' with args {args}: {raw_response}")
    except Exception as e:
        logger.error(f"Exception in list_kubernetes_namespaces with args {args}: {e}")

    session_vars[save_as] = df_result
    return df_result

@app.tool()
@log_tool
async def get_pod(name: str, namespace: str, *, save_as: str) -> pd.DataFrame:
    """
    Retrieves details for a specific Kubernetes Pod by its name and namespace.

    This tool wraps the 'pods_get' tool from the external Kubernetes MCP server.
    The details of the specified Pod are returned as a single-row Pandas DataFrame.

    The output DataFrame is **mandatorily** saved to `session_vars` under the key
    specified by `save_as`.

    Parameters
    ----------
    name : str
        The name of the Pod to retrieve.
    namespace : str
        The namespace where the Pod is located.
    save_as : str
        The **required** key under which the resulting Pandas DataFrame (containing
        the Pod's details) will be stored in `session_vars`.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with a single row representing the Pod, including attributes
        like name, namespace, status, IP, node, and container information.
        Returns an empty DataFrame if the Pod is not found or an error occurs.

    Side Effects
    ------------
    - Stores the output DataFrame in `session_vars` (key: `save_as`).
    - Logs information about the operation success or failure.
    """
    args = {"name": name, "namespace": namespace}
    tool_name = "pods_get"
    df_result = pd.DataFrame()
    try:
        raw_response = await _call_kubernetes_mcp_tool(tool_name, args)
        if raw_response and isinstance(raw_response, dict) and "pod" in raw_response:
            df_result = pd.DataFrame([raw_response["pod"]])
            logger.info(f"Successfully retrieved Pod '{name}' in namespace '{namespace}'. Saved to '{save_as}'.")
        elif raw_response is None:
            logger.error(f"Failed to get Pod for tool '{tool_name}' with args {args}. SDK call failed.")
        else:
            logger.error(f"Unexpected response structure for tool '{tool_name}' with args {args}: {raw_response}")
    except Exception as e:
        logger.error(f"Exception in get_pod with args {args}: {e}")

    session_vars[save_as] = df_result
    return df_result

@app.tool()
@log_tool
async def list_all_pods(*, save_as: str) -> pd.DataFrame:
    """
    Lists all Kubernetes Pods across all namespaces in the current cluster.

    This tool wraps the 'pods_list' tool from the external Kubernetes MCP server.
    The list of Pods is returned as a Pandas DataFrame.

    The output DataFrame is **mandatorily** saved to `session_vars` under the key
    specified by `save_as`.

    Parameters
    ----------
    save_as : str
        The **required** key under which the resulting Pandas DataFrame (containing
        details of all Pods) will be stored in `session_vars`.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame where each row represents a Pod, including attributes like
        name, namespace, status, IP, and node.
        Returns an empty DataFrame if an error occurs or no Pods are found.

    Side Effects
    ------------
    - Stores the output DataFrame in `session_vars` (key: `save_as`).
    - Logs information about the operation success or failure.
    """
    args = {}
    tool_name = "pods_list"
    df_result = pd.DataFrame()
    try:
        raw_response = await _call_kubernetes_mcp_tool(tool_name, args)
        if raw_response and isinstance(raw_response, dict) and "pods" in raw_response:
            df_result = pd.DataFrame(raw_response["pods"])
            logger.info(f"Successfully listed all Pods. Saved to '{save_as}'.")
        elif raw_response is None:
            logger.error(f"Failed to list all Pods for tool '{tool_name}' with args {args}. SDK call failed.")
        else:
            logger.error(f"Unexpected response structure for tool '{tool_name}' with args {args}: {raw_response}")
    except Exception as e:
        logger.error(f"Exception in list_all_pods with args {args}: {e}")

    session_vars[save_as] = df_result
    return df_result

@app.tool()
@log_tool
async def list_pods_in_namespace(namespace: str, *, save_as: str) -> pd.DataFrame:
    """
    Lists all Kubernetes Pods within a specified namespace.

    This tool wraps the 'pods_list_in_namespace' tool from the external Kubernetes MCP server.
    The list of Pods from the given namespace is returned as a Pandas DataFrame.

    The output DataFrame is **mandatorily** saved to `session_vars` under the key
    specified by `save_as`.

    Parameters
    ----------
    namespace : str
        The Kubernetes namespace from which to list Pods.
    save_as : str
        The **required** key under which the resulting Pandas DataFrame (containing
        details of Pods in the namespace) will be stored in `session_vars`.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame where each row represents a Pod in the specified namespace.
        Returns an empty DataFrame if an error occurs or no Pods are found.

    Side Effects
    ------------
    - Stores the output DataFrame in `session_vars` (key: `save_as`).
    - Logs information about the operation success or failure.
    """
    args = {"namespace": namespace}
    tool_name = "pods_list_in_namespace"
    df_result = pd.DataFrame()
    try:
        raw_response = await _call_kubernetes_mcp_tool(tool_name, args)
        if raw_response and isinstance(raw_response, dict) and "pods" in raw_response:
            df_result = pd.DataFrame(raw_response["pods"])
            logger.info(f"Successfully listed Pods in namespace '{namespace}'. Saved to '{save_as}'.")
        elif raw_response is None:
            logger.error(f"Failed to list Pods in namespace for tool '{tool_name}' with args {args}. SDK call failed.")
        else:
            logger.error(f"Unexpected response structure for tool '{tool_name}' with args {args}: {raw_response}")
    except Exception as e:
        logger.error(f"Exception in list_pods_in_namespace with args {args}: {e}")

    session_vars[save_as] = df_result
    return df_result

@app.tool()
@log_tool
async def get_pod_logs(name: str, namespace: str, container: Optional[str] = None, 
                       since_seconds: Optional[int] = None, tail_lines: Optional[int] = None,
                       previous: Optional[bool] = False, timestamps: Optional[bool] = False, *, save_as: str) -> pd.DataFrame:
    """
    Retrieves logs for a specific Kubernetes Pod.

    This tool wraps the 'pods_log' tool from the external Kubernetes MCP server.
    It fetches logs for the specified Pod and container. The logs are returned as a
    Pandas DataFrame with a single column 'log_line'.

    The `since_seconds`, `tail_lines`, `previous`, and `timestamps` parameters are included
    for API consistency with common log retrieval patterns (like `kubectl logs`), but
    they are not directly passed to the underlying `manusa/kubernetes-mcp-server` 'pods_log'
    tool as it only supports `name`, `namespace`, and `container`.

    The output DataFrame is **mandatorily** saved to `session_vars` under the key
    specified by `save_as`.

    Parameters
    ----------
    name : str
        The name of the Pod.
    namespace : str
        The namespace of the Pod.
    container : Optional[str], default None
        The name of the container within the Pod from which to get logs.
        If None, the MCP server might default to the first container or require it.
    since_seconds : Optional[int], default None
        Not used by the underlying MCP tool. For API consistency.
    tail_lines : Optional[int], default None
        Not used by the underlying MCP tool. For API consistency.
    previous : Optional[bool], default False
        Not used by the underlying MCP tool. For API consistency.
    timestamps : Optional[bool], default False
        Not used by the underlying MCP tool. For API consistency.
    save_as : str
        The **required** key under which the resulting Pandas DataFrame (containing
        log lines) will be stored in `session_vars`.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with a single column 'log_line', where each row is a line
        from the Pod's logs. Returns an empty DataFrame (with 'log_line' column)
        if an error occurs or no logs are retrieved.

    Side Effects
    ------------
    - Stores the output DataFrame in `session_vars` (key: `save_as`).
    - Logs information about the operation success or failure.
    """
    args = {"name": name, "namespace": namespace}
    if container:
        args["container"] = container
    tool_name = "pods_log"
    df_result = pd.DataFrame(columns=["log_line"]) 
    try:
        raw_response = await _call_kubernetes_mcp_tool(tool_name, args)
        if raw_response and isinstance(raw_response, dict) and "logs" in raw_response and isinstance(raw_response["logs"], list):
            df_result = pd.DataFrame(raw_response["logs"], columns=["log_line"])
            logger.info(f"Successfully retrieved logs for Pod '{name}' in namespace '{namespace}'. Saved to '{save_as}'.")
        elif raw_response is None:
            logger.error(f"Failed to get Pod logs for tool '{tool_name}' with args {args}. SDK call failed.")
        else:
            logger.error(f"Unexpected response structure for tool '{tool_name}' with args {args}: {raw_response}")
    except Exception as e:
        logger.error(f"Exception in get_pod_logs with args {args}: {e}")

    session_vars[save_as] = df_result
    return df_result

@app.tool()
@log_tool
async def list_openshift_projects(*, save_as: str) -> pd.DataFrame:
    """
    Lists all OpenShift projects in the current cluster. (OpenShift specific)

    This tool wraps the 'projects_list' tool from the external Kubernetes MCP server,
    which is specific to OpenShift clusters. The projects are returned as a Pandas DataFrame.

    The output DataFrame is **mandatorily** saved to `session_vars` under the key
    specified by `save_as`.

    Parameters
    ----------
    save_as : str
        The **required** key under which the resulting Pandas DataFrame (containing
        OpenShift project details) will be stored in `session_vars`.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame where each row represents an OpenShift project.
        Returns an empty DataFrame if an error occurs or no projects are found.

    Side Effects
    ------------
    - Stores the output DataFrame in `session_vars` (key: `save_as`).
    - Logs information about the operation success or failure.
    """
    args = {}
    tool_name = "projects_list"
    df_result = pd.DataFrame()
    try:
        raw_response = await _call_kubernetes_mcp_tool(tool_name, args)
        if raw_response and isinstance(raw_response, dict) and "projects" in raw_response:
            df_result = pd.DataFrame(raw_response["projects"])
            logger.info(f"Successfully listed OpenShift projects. Saved to '{save_as}'.")
        elif raw_response is None:
            logger.error(f"Failed to list OpenShift projects for tool '{tool_name}' with args {args}. SDK call failed.")
        else:
            logger.error(f"Unexpected response structure for tool '{tool_name}' with args {args}: {raw_response}")
    except Exception as e:
        logger.error(f"Exception in list_openshift_projects with args {args}: {e}")

    session_vars[save_as] = df_result
    return df_result

@app.tool()
@log_tool
async def get_kubernetes_resource(api_version: str, kind: str, name: str, namespace: Optional[str] = None, *, save_as: str) -> pd.DataFrame:
    """
    Retrieves a specific Kubernetes resource by its API version, kind, name, and optionally namespace.

    This tool wraps the 'resources_get' tool from the external Kubernetes MCP server.
    It allows fetching any Kubernetes resource (e.g., Deployment, Service, ConfigMap).
    The fetched resource is returned as a single-row Pandas DataFrame.

    The output DataFrame is **mandatorily** saved to `session_vars` under the key
    specified by `save_as`.

    Parameters
    ----------
    api_version : str
        The API version of the resource (e.g., 'v1', 'apps/v1').
    kind : str
        The kind of the resource (e.g., 'Pod', 'Service', 'Deployment').
    name : str
        The name of the resource.
    namespace : Optional[str], default None
        The namespace of the resource. Required for namespaced resources.
        Ignored for cluster-scoped resources.
    save_as : str
        The **required** key under which the resulting Pandas DataFrame (containing
        the resource details) will be stored in `session_vars`.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with a single row representing the fetched Kubernetes resource.
        The resource's structure is preserved, potentially leading to nested data within cells.
        Returns an empty DataFrame if the resource is not found or an error occurs.

    Side Effects
    ------------
    - Stores the output DataFrame in `session_vars` (key: `save_as`).
    - Logs information about the operation success or failure.
    """
    args = {"apiVersion": api_version, "kind": kind, "name": name}
    if namespace:
        args["namespace"] = namespace
    tool_name = "resources_get"
    df_result = pd.DataFrame()
    try:
        raw_response = await _call_kubernetes_mcp_tool(tool_name, args)
        if raw_response and isinstance(raw_response, dict) and "resource" in raw_response:
            df_result = pd.DataFrame([raw_response["resource"]])
            logger.info(f"Successfully retrieved resource '{kind}/{name}'. Saved to '{save_as}'.")
        elif raw_response is None:
            logger.error(f"Failed to get Kubernetes resource for tool '{tool_name}' with args {args}. SDK call failed.")
        else:
            logger.error(f"Unexpected response structure for tool '{tool_name}' with args {args}: {raw_response}")
    except Exception as e:
        logger.error(f"Exception in get_kubernetes_resource with args {args}: {e}")

    session_vars[save_as] = df_result
    return df_result

@app.tool()
@log_tool
async def list_kubernetes_resources(api_version: str, kind: str, namespace: Optional[str] = None, *, save_as: str) -> pd.DataFrame:
    """
    Lists Kubernetes resources of a specific API version and kind.

    This tool wraps the 'resources_list' tool from the external Kubernetes MCP server.
    It can list resources from a specific namespace or from all namespaces if none is provided
    (for namespaced resources). For cluster-scoped resources, the namespace is ignored.
    The resources are returned as a Pandas DataFrame.

    The output DataFrame is **mandatorily** saved to `session_vars` under the key
    specified by `save_as`.

    Parameters
    ----------
    api_version : str
        The API version of the resources to list (e.g., 'v1', 'apps/v1').
    kind : str
        The kind of the resources to list (e.g., 'Pod', 'Service', 'Deployment').
    namespace : Optional[str], default None
        The namespace from which to list resources. If None and the resource is namespaced,
        lists from all namespaces. Ignored for cluster-scoped resources.
    save_as : str
        The **required** key under which the resulting Pandas DataFrame (containing
        the list of resources) will be stored in `session_vars`.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame where each row represents a Kubernetes resource of the specified
        type. Returns an empty DataFrame if an error occurs or no resources are found.

    Side Effects
    ------------
    - Stores the output DataFrame in `session_vars` (key: `save_as`).
    - Logs information about the operation success or failure.
    """
    args = {"apiVersion": api_version, "kind": kind}
    if namespace:
        args["namespace"] = namespace
    tool_name = "resources_list"
    df_result = pd.DataFrame()
    try:
        raw_response = await _call_kubernetes_mcp_tool(tool_name, args)
        if raw_response and isinstance(raw_response, dict) and "resources" in raw_response:
            df_result = pd.DataFrame(raw_response["resources"])
            logger.info(f"Successfully listed resources '{kind}' in namespace '{namespace if namespace else 'all'}'. Saved to '{save_as}'.")
        elif raw_response is None:
            logger.error(f"Failed to list Kubernetes resources for tool '{tool_name}' with args {args}. SDK call failed.")
        else:
            logger.error(f"Unexpected response structure for tool '{tool_name}' with args {args}: {raw_response}")
    except Exception as e:
        logger.error(f"Exception in list_kubernetes_resources with args {args}: {e}")

    session_vars[save_as] = df_result
    return df_result
