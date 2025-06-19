"""Kubernetes Tools for Sherlog MCP Server

This module provides tools for read-only interactions with Kubernetes clusters.
All operations are logged and can be accessed through audit endpoints.
"""

import json
from datetime import datetime

import pandas as pd
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from sherlog_mcp.config import get_settings
from sherlog_mcp.dataframe_utils import smart_create_dataframe, to_pandas
from sherlog_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from sherlog_mcp.session import app, logger


def _kubernetes_available() -> bool:
    """Check if Kubernetes cluster is accessible."""
    try:
        settings = get_settings()

        if hasattr(settings, "kubeconfig_path") and settings.kubeconfig_path:
            config.load_kube_config(config_file=settings.kubeconfig_path)
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()

        v1 = client.CoreV1Api()
        v1.list_namespace(limit=1)
        return True
    except Exception:
        return False


def _get_k8s_client():
    """Get configured Kubernetes client."""
    try:
        settings = get_settings()

        if hasattr(settings, "kubeconfig_path") and settings.kubeconfig_path:
            config.load_kube_config(config_file=settings.kubeconfig_path)
            logger.info(f"Loaded kubeconfig from: {settings.kubeconfig_path}")
        else:
            try:
                config.load_incluster_config()
                logger.info("Using in-cluster Kubernetes configuration")
            except config.ConfigException:
                config.load_kube_config()
                logger.info("Using default kubeconfig from ~/.kube/config")

        v1 = client.CoreV1Api()
        apps_v1 = client.AppsV1Api()

        v1.list_namespace(limit=1)
        logger.info("Successfully connected to Kubernetes cluster")

        return v1, apps_v1

    except Exception as e:
        error_msg = f"""
        Failed to connect to Kubernetes cluster: {str(e)}
        
        Please ensure:
        1. kubectl is configured and working
        2. You have access to a Kubernetes cluster
        3. Set KUBECONFIG_PATH in environment if using custom kubeconfig
        4. Or run from within a Kubernetes cluster with proper service account
        """
        logger.error(error_msg)
        raise Exception(error_msg)


def _list_namespaces_impl() -> pd.DataFrame:
    """List all namespaces in the Kubernetes cluster.

    Returns:
        pd.DataFrame: Namespaces data as a DataFrame

    """
    try:
        v1, _ = _get_k8s_client()

        logger.info("Listing Kubernetes namespaces")
        namespaces = v1.list_namespace()

        if not namespaces.items:
            return pd.DataFrame(
                columns=["name", "status", "age_days", "labels", "annotations"]
            )

        rows = []
        for ns in namespaces.items:
            created = ns.metadata.creation_timestamp
            age_days = (
                (datetime.now(created.tzinfo) - created).days if created else None
            )

            row = {
                "name": ns.metadata.name,
                "status": ns.status.phase,
                "age_days": age_days,
                "labels": json.dumps(ns.metadata.labels or {}),
                "annotations": json.dumps(ns.metadata.annotations or {}),
                "uid": ns.metadata.uid,
                "creation_timestamp": created.isoformat() if created else None,
                "resource_version": ns.metadata.resource_version,
            }
            rows.append(row)

        logger.info(f"Retrieved {len(rows)} namespaces")
        return pd.DataFrame(rows)

    except ApiException as e:
        logger.error(f"Kubernetes API error: {e}")
        raise Exception(f"Failed to list namespaces: {e}")
    except Exception as e:
        logger.error(f"Error listing namespaces: {e}")
        raise


def _list_pods_impl(
    namespace: str = "default", label_selector: str | None = None
) -> pd.DataFrame:
    """List pods in a specific namespace or all namespaces.

    Args:
        namespace (str): Namespace to list pods from, use "all" for all namespaces
        label_selector (Optional[str]): Label selector to filter pods

    Returns:
        pd.DataFrame: Pods data as a DataFrame

    """
    try:
        v1, _ = _get_k8s_client()

        logger.info(f"Listing Kubernetes pods in namespace: {namespace}")

        if namespace == "all":
            pods = v1.list_pod_for_all_namespaces(label_selector=label_selector)
        else:
            pods = v1.list_namespaced_pod(
                namespace=namespace, label_selector=label_selector
            )

        if not pods.items:
            return pd.DataFrame(
                columns=[
                    "name",
                    "namespace",
                    "status",
                    "ready",
                    "restarts",
                    "age_days",
                    "node",
                ]
            )

        rows = []
        for pod in pods.items:
            created = pod.metadata.creation_timestamp
            age_days = (
                (datetime.now(created.tzinfo) - created).days if created else None
            )

            # Calculate ready containers
            ready_count = 0
            total_count = len(pod.status.container_statuses or [])
            if pod.status.container_statuses:
                ready_count = sum(1 for cs in pod.status.container_statuses if cs.ready)

            # Calculate restart count
            restart_count = 0
            if pod.status.container_statuses:
                restart_count = sum(
                    cs.restart_count for cs in pod.status.container_statuses
                )

            row = {
                "name": pod.metadata.name,
                "namespace": pod.metadata.namespace,
                "status": pod.status.phase,
                "ready": f"{ready_count}/{total_count}",
                "restarts": restart_count,
                "age_days": age_days,
                "node": pod.spec.node_name,
                "pod_ip": pod.status.pod_ip,
                "host_ip": pod.status.host_ip,
                "labels": json.dumps(pod.metadata.labels or {}),
                "annotations": json.dumps(pod.metadata.annotations or {}),
                "uid": pod.metadata.uid,
                "creation_timestamp": created.isoformat() if created else None,
                "resource_version": pod.metadata.resource_version,
            }
            rows.append(row)

        logger.info(f"Retrieved {len(rows)} pods")
        # Use smart DataFrame creation for better performance with polars when available
        df = smart_create_dataframe(rows, prefer_polars=True)
        return to_pandas(df)

    except ApiException as e:
        logger.error(f"Kubernetes API error: {e}")
        raise Exception(f"Failed to list pods: {e}")
    except Exception as e:
        logger.error(f"Error listing pods: {e}")
        raise


def _get_pod_logs_impl(
    pod_name: str,
    namespace: str = "default",
    container: str | None = None,
    tail_lines: int = 100,
) -> str:
    """Get logs from a specific pod.

    Args:
        pod_name (str): Name of the pod
        namespace (str): Namespace of the pod
        container (Optional[str]): Specific container name (if pod has multiple containers)
        tail_lines (int): Number of log lines to retrieve from the end

    Returns:
        str: Pod logs

    """
    try:
        v1, _ = _get_k8s_client()

        logger.info(f"Getting logs for pod {pod_name} in namespace {namespace}")

        kwargs = {
            "name": pod_name,
            "namespace": namespace,
            "tail_lines": tail_lines,
            "timestamps": True,
        }

        if container:
            kwargs["container"] = container

        logs = v1.read_namespaced_pod_log(**kwargs)

        logger.info(
            f"Retrieved {len(logs.splitlines())} lines of logs for pod {pod_name}"
        )
        return logs

    except ApiException as e:
        logger.error(f"Kubernetes API error: {e}")
        raise Exception(f"Failed to get pod logs: {e}")
    except Exception as e:
        logger.error(f"Error getting pod logs: {e}")
        raise


def _list_services_impl(namespace: str = "default") -> pd.DataFrame:
    """List services in a specific namespace or all namespaces.

    Args:
        namespace (str): Namespace to list services from, use "all" for all namespaces

    Returns:
        pd.DataFrame: Services data as a DataFrame

    """
    try:
        v1, _ = _get_k8s_client()

        logger.info(f"Listing Kubernetes services in namespace: {namespace}")

        if namespace == "all":
            services = v1.list_service_for_all_namespaces()
        else:
            services = v1.list_namespaced_service(namespace=namespace)

        if not services.items:
            return pd.DataFrame(
                columns=[
                    "name",
                    "namespace",
                    "type",
                    "cluster_ip",
                    "external_ip",
                    "ports",
                    "age_days",
                ]
            )

        rows = []
        for svc in services.items:
            created = svc.metadata.creation_timestamp
            age_days = (
                (datetime.now(created.tzinfo) - created).days if created else None
            )

            # Format ports
            ports = []
            if svc.spec.ports:
                for port in svc.spec.ports:
                    port_str = f"{port.port}"
                    if port.target_port:
                        port_str += f":{port.target_port}"
                    if port.protocol:
                        port_str += f"/{port.protocol}"
                    ports.append(port_str)

            # Format external IPs
            external_ips = []
            if svc.status.load_balancer and svc.status.load_balancer.ingress:
                for ingress in svc.status.load_balancer.ingress:
                    if ingress.ip:
                        external_ips.append(ingress.ip)
                    elif ingress.hostname:
                        external_ips.append(ingress.hostname)

            row = {
                "name": svc.metadata.name,
                "namespace": svc.metadata.namespace,
                "type": svc.spec.type,
                "cluster_ip": svc.spec.cluster_ip,
                "external_ip": ",".join(external_ips) if external_ips else None,
                "ports": ",".join(ports),
                "age_days": age_days,
                "selector": json.dumps(svc.spec.selector or {}),
                "labels": json.dumps(svc.metadata.labels or {}),
                "annotations": json.dumps(svc.metadata.annotations or {}),
                "uid": svc.metadata.uid,
                "creation_timestamp": created.isoformat() if created else None,
            }
            rows.append(row)

        logger.info(f"Retrieved {len(rows)} services")
        return pd.DataFrame(rows)

    except ApiException as e:
        logger.error(f"Kubernetes API error: {e}")
        raise Exception(f"Failed to list services: {e}")
    except Exception as e:
        logger.error(f"Error listing services: {e}")
        raise


def _list_deployments_impl(namespace: str = "default") -> pd.DataFrame:
    """List deployments in a specific namespace or all namespaces.

    Args:
        namespace (str): Namespace to list deployments from, use "all" for all namespaces

    Returns:
        pd.DataFrame: Deployments data as a DataFrame

    """
    try:
        _, apps_v1 = _get_k8s_client()

        logger.info(f"Listing Kubernetes deployments in namespace: {namespace}")

        if namespace == "all":
            deployments = apps_v1.list_deployment_for_all_namespaces()
        else:
            deployments = apps_v1.list_namespaced_deployment(namespace=namespace)

        if not deployments.items:
            return pd.DataFrame(
                columns=[
                    "name",
                    "namespace",
                    "ready",
                    "up_to_date",
                    "available",
                    "age_days",
                ]
            )

        rows = []
        for deploy in deployments.items:
            created = deploy.metadata.creation_timestamp
            age_days = (
                (datetime.now(created.tzinfo) - created).days if created else None
            )

            row = {
                "name": deploy.metadata.name,
                "namespace": deploy.metadata.namespace,
                "ready": f"{deploy.status.ready_replicas or 0}/{deploy.spec.replicas or 0}",
                "up_to_date": deploy.status.updated_replicas or 0,
                "available": deploy.status.available_replicas or 0,
                "age_days": age_days,
                "strategy": deploy.spec.strategy.type if deploy.spec.strategy else None,
                "labels": json.dumps(deploy.metadata.labels or {}),
                "annotations": json.dumps(deploy.metadata.annotations or {}),
                "selector": json.dumps(deploy.spec.selector.match_labels or {}),
                "uid": deploy.metadata.uid,
                "creation_timestamp": created.isoformat() if created else None,
                "generation": deploy.metadata.generation,
                "observed_generation": deploy.status.observed_generation,
            }
            rows.append(row)

        logger.info(f"Retrieved {len(rows)} deployments")
        return pd.DataFrame(rows)

    except ApiException as e:
        logger.error(f"Kubernetes API error: {e}")
        raise Exception(f"Failed to list deployments: {e}")
    except Exception as e:
        logger.error(f"Error listing deployments: {e}")
        raise


def _list_events_impl(namespace: str = "default", limit: int = 100) -> pd.DataFrame:
    """List recent events in a specific namespace or all namespaces.

    Args:
        namespace (str): Namespace to list events from, use "all" for all namespaces
        limit (int): Maximum number of events to retrieve

    Returns:
        pd.DataFrame: Events data as a DataFrame

    """
    try:
        v1, _ = _get_k8s_client()

        logger.info(f"Listing Kubernetes events in namespace: {namespace}")

        if namespace == "all":
            events = v1.list_event_for_all_namespaces(limit=limit)
        else:
            events = v1.list_namespaced_event(namespace=namespace, limit=limit)

        if not events.items:
            return pd.DataFrame(
                columns=[
                    "namespace",
                    "last_seen",
                    "type",
                    "reason",
                    "object",
                    "message",
                ]
            )

        rows = []
        for event in events.items:
            # Sort by last_timestamp to get most recent events first
            last_seen = event.last_timestamp or event.first_timestamp

            object_ref = ""
            if event.involved_object:
                object_ref = (
                    f"{event.involved_object.kind}/{event.involved_object.name}"
                )

            row = {
                "namespace": event.namespace,
                "last_seen": last_seen.isoformat() if last_seen else None,
                "type": event.type,
                "reason": event.reason,
                "object": object_ref,
                "message": event.message,
                "count": event.count,
                "first_timestamp": event.first_timestamp.isoformat()
                if event.first_timestamp
                else None,
                "source_component": event.source.component if event.source else None,
                "source_host": event.source.host if event.source else None,
                "uid": event.metadata.uid,
                "creation_timestamp": event.metadata.creation_timestamp.isoformat()
                if event.metadata.creation_timestamp
                else None,
            }
            rows.append(row)

        # Sort by last_seen (most recent first)
        df = pd.DataFrame(rows)
        if not df.empty and "last_seen" in df.columns:
            df = df.sort_values("last_seen", ascending=False, na_position="last")

        logger.info(f"Retrieved {len(rows)} events")
        return df

    except ApiException as e:
        logger.error(f"Kubernetes API error: {e}")
        raise Exception(f"Failed to list events: {e}")
    except Exception as e:
        logger.error(f"Error listing events: {e}")
        raise


def _get_pod_details_impl(pod_name: str, namespace: str = "default") -> pd.DataFrame:
    """Get detailed information about a specific pod.

    Args:
        pod_name (str): Name of the pod
        namespace (str): Namespace of the pod

    Returns:
        pd.DataFrame: Detailed pod information as a DataFrame

    """
    try:
        v1, _ = _get_k8s_client()

        logger.info(f"Getting details for pod {pod_name} in namespace {namespace}")
        response = v1.read_namespaced_pod(name=pod_name, namespace=namespace)

        # Ensure we have a proper pod object
        if isinstance(response, str):
            raise Exception(f"Unexpected response type: {type(response)}")

        pod = response  # type: ignore
        created = pod.metadata.creation_timestamp  # type: ignore
        age_days = (datetime.now(created.tzinfo) - created).days if created else None

        # Container information
        containers_info = []
        if pod.spec.containers:  # type: ignore
            for container in pod.spec.containers:  # type: ignore
                container_info = {
                    "name": container.name,
                    "image": container.image,
                    "image_pull_policy": container.image_pull_policy,
                    "ports": [
                        f"{p.container_port}/{p.protocol}"
                        for p in (container.ports or [])
                    ],
                    "resources_requests": container.resources.requests
                    if container.resources
                    else None,
                    "resources_limits": container.resources.limits
                    if container.resources
                    else None,
                }
                containers_info.append(container_info)

        # Container status information
        container_statuses = []
        if pod.status.container_statuses:  # type: ignore
            for status in pod.status.container_statuses:  # type: ignore
                status_info = {
                    "name": status.name,
                    "ready": status.ready,
                    "restart_count": status.restart_count,
                    "image": status.image,
                    "state": status.state,
                }
                container_statuses.append(status_info)

        # Conditions
        conditions = []
        if pod.status.conditions:  # type: ignore
            for condition in pod.status.conditions:  # type: ignore
                conditions.append(
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "last_transition_time": condition.last_transition_time.isoformat()
                        if condition.last_transition_time
                        else None,
                        "reason": condition.reason,
                        "message": condition.message,
                    }
                )

        row = {
            "name": pod.metadata.name,  # type: ignore
            "namespace": pod.metadata.namespace,  # type: ignore
            "uid": pod.metadata.uid,  # type: ignore
            "node": pod.spec.node_name,  # type: ignore
            "pod_ip": pod.status.pod_ip,  # type: ignore
            "host_ip": pod.status.host_ip,  # type: ignore
            "phase": pod.status.phase,  # type: ignore
            "qos_class": pod.status.qos_class,  # type: ignore
            "restart_policy": pod.spec.restart_policy,  # type: ignore
            "service_account": pod.spec.service_account_name,  # type: ignore
            "age_days": age_days,
            "containers": json.dumps(containers_info),
            "container_statuses": json.dumps(container_statuses),
            "conditions": json.dumps(conditions),
            "labels": json.dumps(pod.metadata.labels or {}),  # type: ignore
            "annotations": json.dumps(pod.metadata.annotations or {}),  # type: ignore
            "creation_timestamp": created.isoformat() if created else None,
            "start_time": pod.status.start_time.isoformat()  # type: ignore
            if pod.status.start_time  # type: ignore
            else None,  # type: ignore
        }

        logger.info(f"Retrieved details for pod {pod_name}")
        return pd.DataFrame([row])

    except ApiException as e:
        logger.error(f"Kubernetes API error: {e}")
        raise Exception(f"Failed to get pod details: {e}")
    except Exception as e:
        logger.error(f"Error getting pod details: {e}")
        raise


def _list_nodes_impl() -> pd.DataFrame:
    """List all nodes in the Kubernetes cluster.

    Returns:
        pd.DataFrame: Nodes data as a DataFrame

    """
    try:
        v1, _ = _get_k8s_client()

        logger.info("Listing Kubernetes nodes")
        nodes = v1.list_node()

        if not nodes.items:
            return pd.DataFrame(
                columns=[
                    "name",
                    "status",
                    "roles",
                    "age_days",
                    "version",
                    "os",
                    "kernel",
                ]
            )

        rows = []
        for node in nodes.items:
            created = node.metadata.creation_timestamp
            age_days = (
                (datetime.now(created.tzinfo) - created).days if created else None
            )

            # Determine node roles
            roles = []
            if node.metadata.labels:
                for label_key in node.metadata.labels:
                    if label_key.startswith("node-role.kubernetes.io/"):
                        role = label_key.replace("node-role.kubernetes.io/", "")
                        roles.append(role)

            # Get node status
            status = "Unknown"
            if node.status.conditions:
                for condition in node.status.conditions:
                    if condition.type == "Ready":
                        status = "Ready" if condition.status == "True" else "NotReady"
                        break

            row = {
                "name": node.metadata.name,
                "status": status,
                "roles": ",".join(roles) if roles else "worker",
                "age_days": age_days,
                "version": node.status.node_info.kubelet_version
                if node.status.node_info
                else None,
                "os": node.status.node_info.operating_system
                if node.status.node_info
                else None,
                "kernel": node.status.node_info.kernel_version
                if node.status.node_info
                else None,
                "container_runtime": node.status.node_info.container_runtime_version
                if node.status.node_info
                else None,
                "architecture": node.status.node_info.architecture
                if node.status.node_info
                else None,
                "labels": json.dumps(node.metadata.labels or {}),
                "annotations": json.dumps(node.metadata.annotations or {}),
                "uid": node.metadata.uid,
                "creation_timestamp": created.isoformat() if created else None,
            }
            rows.append(row)

        logger.info(f"Retrieved {len(rows)} nodes")
        return pd.DataFrame(rows)

    except ApiException as e:
        logger.error(f"Kubernetes API error: {e}")
        raise Exception(f"Failed to list nodes: {e}")
    except Exception as e:
        logger.error(f"Error listing nodes: {e}")
        raise


# Push implementation functions to shell
_SHELL.push(
    {
        "list_namespaces_impl": _list_namespaces_impl,
        "list_pods_impl": _list_pods_impl,
        "get_pod_logs_impl": _get_pod_logs_impl,
        "list_services_impl": _list_services_impl,
        "list_deployments_impl": _list_deployments_impl,
        "list_events_impl": _list_events_impl,
        "get_pod_details_impl": _get_pod_details_impl,
        "list_nodes_impl": _list_nodes_impl,
    }
)


# Conditional tool registration based on Kubernetes availability
if _kubernetes_available():
    logger.info("Kubernetes cluster detected - registering Kubernetes tools")

    # MCP Tool implementations

    @app.tool()
    async def list_namespaces(*, save_as: str) -> pd.DataFrame | None:
        """List all namespaces in the Kubernetes cluster.

        Args:
            save_as (str): Variable name to store the namespaces data

        Returns:
            pd.DataFrame: Namespaces data as a DataFrame

        """
        code = f"{save_as} = list_namespaces_impl()\n{save_as}"
        execution_result = await run_code_in_shell(code)
        df = execution_result.result if execution_result else None
        if isinstance(df, pd.DataFrame):
            return df.to_dict("records")

    @app.tool()
    async def list_pods(
        namespace: str = "default", label_selector: str | None = None, *, save_as: str
    ) -> pd.DataFrame | None:
        """List pods in a specific namespace or all namespaces.

        Args:
            namespace (str): Namespace to list pods from, use "all" for all namespaces
            label_selector (Optional[str]): Label selector to filter pods (e.g., "app=nginx")
            save_as (str): Variable name to store the pods data

        Returns:
            pd.DataFrame: Pods data as a DataFrame

        """
        code = f'{save_as} = list_pods_impl("{namespace}"'
        if label_selector:
            code += f', "{label_selector}"'
        code += f")\n{save_as}"

        execution_result = await run_code_in_shell(code)
        df = execution_result.result if execution_result else None
        if isinstance(df, pd.DataFrame):
            return df.to_dict("records")

    @app.tool()
    async def get_pod_logs(
        pod_name: str,
        namespace: str = "default",
        container: str | None = None,
        tail_lines: int = 100,
        *,
        save_as: str,
    ) -> str | None:
        """Get logs from a specific pod.

        Args:
            pod_name (str): Name of the pod
            namespace (str): Namespace of the pod
            container (Optional[str]): Specific container name (if pod has multiple containers)
            tail_lines (int): Number of log lines to retrieve from the end
            save_as (str): Variable name to store the logs

        Returns:
            str: Pod logs

        """
        code = f'{save_as} = get_pod_logs_impl("{pod_name}", "{namespace}"'
        if container:
            code += f', "{container}"'
        else:
            code += ", None"
        code += f", {tail_lines})\n{save_as}"

        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    @app.tool()
    async def list_services(
        namespace: str = "default", *, save_as: str
    ) -> pd.DataFrame | None:
        """List services in a specific namespace or all namespaces.

        Args:
            namespace (str): Namespace to list services from, use "all" for all namespaces
            save_as (str): Variable name to store the services data

        Returns:
            pd.DataFrame: Services data as a DataFrame

        """
        code = f'{save_as} = list_services_impl("{namespace}")\n{save_as}'
        execution_result = await run_code_in_shell(code)
        df = execution_result.result if execution_result else None
        if isinstance(df, pd.DataFrame):
            return df.to_dict("records")

    @app.tool()
    async def list_deployments(
        namespace: str = "default", *, save_as: str
    ) -> pd.DataFrame | None:
        """List deployments in a specific namespace or all namespaces.

        Args:
            namespace (str): Namespace to list deployments from, use "all" for all namespaces
            save_as (str): Variable name to store the deployments data

        Returns:
            pd.DataFrame: Deployments data as a DataFrame

        """
        code = f'{save_as} = list_deployments_impl("{namespace}")\n{save_as}'
        execution_result = await run_code_in_shell(code)
        df = execution_result.result if execution_result else None
        if isinstance(df, pd.DataFrame):
            return df.to_dict("records")

    @app.tool()
    async def list_events(
        namespace: str = "default", limit: int = 100, *, save_as: str
    ) -> pd.DataFrame | None:
        """List recent events in a specific namespace or all namespaces.

        Args:
            namespace (str): Namespace to list events from, use "all" for all namespaces
            limit (int): Maximum number of events to retrieve
            save_as (str): Variable name to store the events data

        Returns:
            pd.DataFrame: Events data as a DataFrame

        """
        code = f'{save_as} = list_events_impl("{namespace}", {limit})\n{save_as}'
        execution_result = await run_code_in_shell(code)
        df = execution_result.result if execution_result else None
        if isinstance(df, pd.DataFrame):
            return df.to_dict("records")

    @app.tool()
    async def get_pod_details(
        pod_name: str, namespace: str = "default", *, save_as: str
    ) -> pd.DataFrame | None:
        """Get detailed information about a specific pod.

        Args:
            pod_name (str): Name of the pod
            namespace (str): Namespace of the pod
            save_as (str): Variable name to store the pod details

        Returns:
            pd.DataFrame: Detailed pod information as a DataFrame

        """
        code = (
            f'{save_as} = get_pod_details_impl("{pod_name}", "{namespace}")\n{save_as}'
        )
        execution_result = await run_code_in_shell(code)
        df = execution_result.result if execution_result else None
        if isinstance(df, pd.DataFrame):
            return df.to_dict("records")

    @app.tool()
    async def list_nodes(*, save_as: str) -> pd.DataFrame | None:
        """List all nodes in the Kubernetes cluster.

        Args:
            save_as (str): Variable name to store the nodes data

        Returns:
            pd.DataFrame: Nodes data as a DataFrame

        """
        code = f"{save_as} = list_nodes_impl()\n{save_as}"
        execution_result = await run_code_in_shell(code)
        df = execution_result.result if execution_result else None
        if isinstance(df, pd.DataFrame):
            return df.to_dict("records")

else:
    logger.info(
        "Kubernetes cluster not detected - Kubernetes tools will not be registered"
    )
