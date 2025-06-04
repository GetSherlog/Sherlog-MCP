from typing import Optional
import requests
import pandas as pd
from urllib.parse import urljoin

from logai_mcp.session import (
    app,
    logger,
)

from logai_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from logai_mcp.config import get_settings


def _get_grafana_session():
    """Create a requests session with Grafana authentication headers."""
    settings = get_settings()
    if not settings.grafana_url or not settings.grafana_api_key:
        raise ValueError("GRAFANA_URL and GRAFANA_API_KEY must be set in environment variables")
    
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {settings.grafana_api_key}",
        "Content-Type": "application/json"
    })
    return session, settings.grafana_url


# Prometheus Tools
def _query_prometheus_impl(datasource_uid: str, query: str, query_type: str = "instant", 
                          start: Optional[str] = None, end: Optional[str] = None, 
                          step: Optional[str] = None) -> pd.DataFrame:
    """
    Execute a PromQL query against a Prometheus datasource in Grafana.
    
    Args:
        datasource_uid (str): UID of the Prometheus datasource
        query (str): PromQL query string
        query_type (str): Type of query - "instant" or "range"
        start (Optional[str]): Start time for range queries (RFC3339 format)
        end (Optional[str]): End time for range queries (RFC3339 format)
        step (Optional[str]): Query resolution step width for range queries
        
    Returns:
        pd.DataFrame: Query results from Prometheus as a DataFrame
    """
    session, grafana_url = _get_grafana_session()
    
    endpoint = f"/api/ds/query"
    url = urljoin(grafana_url, endpoint)
    
    if query_type == "range":
        targets = [{
            "expr": query,
            "datasource": {"type": "prometheus", "uid": datasource_uid},
            "format": "time_series",
            "start": start,
            "end": end,
            "step": step
        }]
    else:
        targets = [{
            "expr": query,
            "datasource": {"type": "prometheus", "uid": datasource_uid},
            "format": "time_series",
            "instant": True
        }]
    
    payload = {
        "queries": targets
    }
    
    logger.info(f"Executing Prometheus query: {query}")
    response = session.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    logger.info(f"Prometheus query completed successfully")
    
    # Convert result to DataFrame
    rows = []
    if 'results' in result:
        for query_result in result['results'].values():
            if 'frames' in query_result:
                for frame in query_result['frames']:
                    if 'data' in frame and 'values' in frame['data']:
                        values = frame['data']['values']
                        if len(values) >= 2:  # timestamp and value columns
                            timestamps = values[0] if values[0] else []
                            metric_values = values[1] if len(values) > 1 and values[1] else []
                            
                            # Get metric labels from frame schema
                            labels = {}
                            if 'schema' in frame and 'fields' in frame['schema']:
                                for field in frame['schema']['fields']:
                                    if field.get('name') == 'Value' and 'labels' in field:
                                        labels = field['labels']
                            
                            for i, (ts, val) in enumerate(zip(timestamps, metric_values)):
                                row = {
                                    'timestamp': pd.to_datetime(ts, unit='ms') if ts else None,
                                    'value': val,
                                    'query': query,
                                    'datasource_uid': datasource_uid
                                }
                                # Add labels as separate columns
                                row.update(labels)
                                rows.append(row)
    
    if not rows:
        # Return empty DataFrame with standard columns if no data
        return pd.DataFrame(columns=['timestamp', 'value', 'query', 'datasource_uid'])
    
    return pd.DataFrame(rows)

_SHELL.push({"query_prometheus_impl": _query_prometheus_impl})


@app.tool()
async def query_prometheus(datasource_uid: str, query: str, query_type: str = "instant",
                          start: Optional[str] = None, end: Optional[str] = None,
                          step: Optional[str] = None, *, save_as: str) -> Optional[pd.DataFrame]:
    """
    Execute a PromQL query against a Prometheus datasource.
    
    Args:
        datasource_uid (str): UID of the Prometheus datasource
        query (str): PromQL query string
        query_type (str): Type of query - "instant" or "range" (default: "instant")
        start (Optional[str]): Start time for range queries (RFC3339 format)
        end (Optional[str]): End time for range queries (RFC3339 format)
        step (Optional[str]): Query resolution step width for range queries
        save_as (str): Variable name to store the query results
        
    Returns:
        pd.DataFrame: Query results from Prometheus as a DataFrame
    """
    code = f'{save_as} = query_prometheus_impl("{datasource_uid}", "{query}", "{query_type}"'
    if start:
        code += f', "{start}"'
        if end:
            code += f', "{end}"'
            if step:
                code += f', "{step}"'
    code += f')\n{save_as}'
    
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


def _list_prometheus_metric_metadata_impl(datasource_uid: str, metric: Optional[str] = None) -> pd.DataFrame:
    """
    List metric metadata from a Prometheus datasource.
    
    Args:
        datasource_uid (str): UID of the Prometheus datasource
        metric (Optional[str]): Specific metric name to get metadata for
        
    Returns:
        pd.DataFrame: Metric metadata from Prometheus as a DataFrame
    """
    session, grafana_url = _get_grafana_session()
    
    endpoint = f"/api/datasources/proxy/{datasource_uid}/api/v1/metadata"
    url = urljoin(grafana_url, endpoint)
    
    params = {}
    if metric:
        params["metric"] = metric
    
    logger.info(f"Fetching Prometheus metric metadata for datasource: {datasource_uid}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    result = response.json()
    logger.info(f"Retrieved metadata for {len(result.get('data', {}))} metrics")
    
    # Convert metadata to DataFrame
    rows = []
    metadata = result.get('data', {})
    for metric_name, metric_info in metadata.items():
        for entry in metric_info:
            row = {
                'metric_name': metric_name,
                'type': entry.get('type', ''),
                'help': entry.get('help', ''),
                'unit': entry.get('unit', ''),
                'datasource_uid': datasource_uid
            }
            rows.append(row)
    
    if not rows:
        return pd.DataFrame(columns=['metric_name', 'type', 'help', 'unit', 'datasource_uid'])
    
    return pd.DataFrame(rows)

_SHELL.push({"list_prometheus_metric_metadata_impl": _list_prometheus_metric_metadata_impl})


@app.tool()
async def list_prometheus_metric_metadata(datasource_uid: str, metric: Optional[str] = None, 
                                        *, save_as: str) -> Optional[pd.DataFrame]:
    """
    List metric metadata from a Prometheus datasource.
    
    Args:
        datasource_uid (str): UID of the Prometheus datasource
        metric (Optional[str]): Specific metric name to get metadata for
        save_as (str): Variable name to store the metadata
        
    Returns:
        pd.DataFrame: Metric metadata from Prometheus as a DataFrame
    """
    if metric:
        code = f'{save_as} = list_prometheus_metric_metadata_impl("{datasource_uid}", "{metric}")\n{save_as}'
    else:
        code = f'{save_as} = list_prometheus_metric_metadata_impl("{datasource_uid}")\n{save_as}'
    
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


def _list_prometheus_metric_names_impl(datasource_uid: str) -> pd.DataFrame:
    """
    List available metric names from a Prometheus datasource.
    
    Args:
        datasource_uid (str): UID of the Prometheus datasource
        
    Returns:
        pd.DataFrame: DataFrame with metric names
    """
    session, grafana_url = _get_grafana_session()
    
    endpoint = f"/api/datasources/proxy/{datasource_uid}/api/v1/label/__name__/values"
    url = urljoin(grafana_url, endpoint)
    
    logger.info(f"Fetching Prometheus metric names for datasource: {datasource_uid}")
    response = session.get(url)
    response.raise_for_status()
    
    result = response.json()
    metric_names = result.get("data", [])
    logger.info(f"Retrieved {len(metric_names)} metric names")
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'metric_name': metric_names,
        'datasource_uid': [datasource_uid] * len(metric_names)
    })
    
    return df

_SHELL.push({"list_prometheus_metric_names_impl": _list_prometheus_metric_names_impl})


@app.tool()
async def list_prometheus_metric_names(datasource_uid: str, *, save_as: str) -> Optional[pd.DataFrame]:
    """
    List available metric names from a Prometheus datasource.
    
    Args:
        datasource_uid (str): UID of the Prometheus datasource
        save_as (str): Variable name to store the metric names
        
    Returns:
        pd.DataFrame: DataFrame with metric names
    """
    code = f'{save_as} = list_prometheus_metric_names_impl("{datasource_uid}")\n{save_as}'
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


def _list_prometheus_label_names_impl(datasource_uid: str, match: Optional[str] = None) -> pd.DataFrame:
    """
    List label names from a Prometheus datasource.
    
    Args:
        datasource_uid (str): UID of the Prometheus datasource
        match (Optional[str]): Series selector to match against
        
    Returns:
        pd.DataFrame: DataFrame with label names
    """
    session, grafana_url = _get_grafana_session()
    
    endpoint = f"/api/datasources/proxy/{datasource_uid}/api/v1/labels"
    url = urljoin(grafana_url, endpoint)
    
    params = {}
    if match:
        params["match[]"] = match
    
    logger.info(f"Fetching Prometheus label names for datasource: {datasource_uid}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    result = response.json()
    label_names = result.get("data", [])
    logger.info(f"Retrieved {len(label_names)} label names")
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'label_name': label_names,
        'datasource_uid': [datasource_uid] * len(label_names),
        'match_selector': [match] * len(label_names) if match else [None] * len(label_names)
    })
    
    return df

_SHELL.push({"list_prometheus_label_names_impl": _list_prometheus_label_names_impl})


@app.tool()
async def list_prometheus_label_names(datasource_uid: str, match: Optional[str] = None, 
                                    *, save_as: str) -> Optional[pd.DataFrame]:
    """
    List label names from a Prometheus datasource.
    
    Args:
        datasource_uid (str): UID of the Prometheus datasource
        match (Optional[str]): Series selector to match against
        save_as (str): Variable name to store the label names
        
    Returns:
        pd.DataFrame: DataFrame with label names
    """
    if match:
        code = f'{save_as} = list_prometheus_label_names_impl("{datasource_uid}", "{match}")\n{save_as}'
    else:
        code = f'{save_as} = list_prometheus_label_names_impl("{datasource_uid}")\n{save_as}'
    
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


def _list_prometheus_label_values_impl(datasource_uid: str, label: str, match: Optional[str] = None) -> pd.DataFrame:
    """
    List values for a specific label from a Prometheus datasource.
    
    Args:
        datasource_uid (str): UID of the Prometheus datasource
        label (str): Label name to get values for
        match (Optional[str]): Series selector to match against
        
    Returns:
        pd.DataFrame: DataFrame with label values
    """
    session, grafana_url = _get_grafana_session()
    
    endpoint = f"/api/datasources/proxy/{datasource_uid}/api/v1/label/{label}/values"
    url = urljoin(grafana_url, endpoint)
    
    params = {}
    if match:
        params["match[]"] = match
    
    logger.info(f"Fetching Prometheus label values for label '{label}' in datasource: {datasource_uid}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    result = response.json()
    label_values = result.get("data", [])
    logger.info(f"Retrieved {len(label_values)} values for label '{label}'")
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'label_name': [label] * len(label_values),
        'label_value': label_values,
        'datasource_uid': [datasource_uid] * len(label_values),
        'match_selector': [match] * len(label_values) if match else [None] * len(label_values)
    })
    
    return df

_SHELL.push({"list_prometheus_label_values_impl": _list_prometheus_label_values_impl})


@app.tool()
async def list_prometheus_label_values(datasource_uid: str, label: str, match: Optional[str] = None,
                                     *, save_as: str) -> Optional[pd.DataFrame]:
    """
    List values for a specific label from a Prometheus datasource.
    
    Args:
        datasource_uid (str): UID of the Prometheus datasource
        label (str): Label name to get values for
        match (Optional[str]): Series selector to match against
        save_as (str): Variable name to store the label values
        
    Returns:
        pd.DataFrame: DataFrame with label values
    """
    if match:
        code = f'{save_as} = list_prometheus_label_values_impl("{datasource_uid}", "{label}", "{match}")\n{save_as}'
    else:
        code = f'{save_as} = list_prometheus_label_values_impl("{datasource_uid}", "{label}")\n{save_as}'
    
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


# Loki Tools
def _query_loki_logs_impl(datasource_uid: str, query: str, query_type: str = "logs",
                         start: Optional[str] = None, end: Optional[str] = None,
                         limit: Optional[int] = None, direction: str = "backward") -> pd.DataFrame:
    """
    Query and retrieve logs using LogQL from a Loki datasource.
    
    Args:
        datasource_uid (str): UID of the Loki datasource
        query (str): LogQL query string
        query_type (str): Type of query - "logs" or "metrics"
        start (Optional[str]): Start time (RFC3339 format)
        end (Optional[str]): End time (RFC3339 format)
        limit (Optional[int]): Maximum number of entries to return
        direction (str): Direction to scan - "forward" or "backward"
        
    Returns:
        pd.DataFrame: Query results from Loki as a DataFrame
    """
    session, grafana_url = _get_grafana_session()
    
    endpoint = f"/api/ds/query"
    url = urljoin(grafana_url, endpoint)
    
    targets = [{
        "expr": query,
        "datasource": {"type": "loki", "uid": datasource_uid},
        "queryType": query_type,
        "maxLines": limit,
        "direction": direction
    }]
    
    if start:
        targets[0]["start"] = start
    if end:
        targets[0]["end"] = end
    
    payload = {
        "queries": targets
    }
    
    logger.info(f"Executing Loki query: {query}")
    response = session.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    logger.info(f"Loki query completed successfully")
    
    # Convert result to DataFrame
    rows = []
    if 'results' in result:
        for query_result in result['results'].values():
            if 'frames' in query_result:
                for frame in query_result['frames']:
                    if 'data' in frame and 'values' in frame['data']:
                        values = frame['data']['values']
                        if len(values) >= 2:  # timestamp and log line columns
                            timestamps = values[0] if values[0] else []
                            log_lines = values[1] if len(values) > 1 and values[1] else []
                            
                            # Get labels from frame schema
                            labels = {}
                            if 'schema' in frame and 'fields' in frame['schema']:
                                for field in frame['schema']['fields']:
                                    if field.get('name') == 'Line' and 'labels' in field:
                                        labels = field['labels']
                            
                            for i, (ts, line) in enumerate(zip(timestamps, log_lines)):
                                row = {
                                    'timestamp': pd.to_datetime(ts, unit='ns') if ts else None,
                                    'log_line': line,
                                    'query': query,
                                    'query_type': query_type,
                                    'datasource_uid': datasource_uid
                                }
                                # Add labels as separate columns
                                row.update(labels)
                                rows.append(row)
    
    if not rows:
        # Return empty DataFrame with standard columns if no data
        return pd.DataFrame(columns=['timestamp', 'log_line', 'query', 'query_type', 'datasource_uid'])
    
    return pd.DataFrame(rows)

_SHELL.push({"query_loki_logs_impl": _query_loki_logs_impl})


@app.tool()
async def query_loki_logs(datasource_uid: str, query: str, query_type: str = "logs",
                         start: Optional[str] = None, end: Optional[str] = None,
                         limit: Optional[int] = None, direction: str = "backward",
                         *, save_as: str) -> Optional[pd.DataFrame]:
    """
    Query and retrieve logs using LogQL from a Loki datasource.
    
    Args:
        datasource_uid (str): UID of the Loki datasource
        query (str): LogQL query string
        query_type (str): Type of query - "logs" or "metrics" (default: "logs")
        start (Optional[str]): Start time (RFC3339 format)
        end (Optional[str]): End time (RFC3339 format)
        limit (Optional[int]): Maximum number of entries to return
        direction (str): Direction to scan - "forward" or "backward" (default: "backward")
        save_as (str): Variable name to store the query results
        
    Returns:
        pd.DataFrame: Query results from Loki as a DataFrame
    """
    code = f'{save_as} = query_loki_logs_impl("{datasource_uid}", "{query}", "{query_type}"'
    if start:
        code += f', "{start}"'
        if end:
            code += f', "{end}"'
            if limit:
                code += f', {limit}'
                if direction != "backward":
                    code += f', "{direction}"'
    code += f')\n{save_as}'
    
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


def _list_loki_label_names_impl(datasource_uid: str, start: Optional[str] = None, 
                               end: Optional[str] = None) -> pd.DataFrame:
    """
    List all available label names in logs from a Loki datasource.
    
    Args:
        datasource_uid (str): UID of the Loki datasource
        start (Optional[str]): Start time (RFC3339 format)
        end (Optional[str]): End time (RFC3339 format)
        
    Returns:
        pd.DataFrame: DataFrame with label names
    """
    session, grafana_url = _get_grafana_session()
    
    endpoint = f"/api/datasources/proxy/{datasource_uid}/loki/api/v1/labels"
    url = urljoin(grafana_url, endpoint)
    
    params = {}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    
    logger.info(f"Fetching Loki label names for datasource: {datasource_uid}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    result = response.json()
    label_names = result.get("data", [])
    logger.info(f"Retrieved {len(label_names)} label names from Loki")
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'label_name': label_names,
        'datasource_uid': [datasource_uid] * len(label_names),
        'start_time': [start] * len(label_names) if start else [None] * len(label_names),
        'end_time': [end] * len(label_names) if end else [None] * len(label_names)
    })
    
    return df

_SHELL.push({"list_loki_label_names_impl": _list_loki_label_names_impl})


@app.tool()
async def list_loki_label_names(datasource_uid: str, start: Optional[str] = None,
                               end: Optional[str] = None, *, save_as: str) -> Optional[pd.DataFrame]:
    """
    List all available label names in logs from a Loki datasource.
    
    Args:
        datasource_uid (str): UID of the Loki datasource
        start (Optional[str]): Start time (RFC3339 format)
        end (Optional[str]): End time (RFC3339 format)
        save_as (str): Variable name to store the label names
        
    Returns:
        pd.DataFrame: DataFrame with label names
    """
    code = f'{save_as} = list_loki_label_names_impl("{datasource_uid}"'
    if start:
        code += f', "{start}"'
        if end:
            code += f', "{end}"'
    code += f')\n{save_as}'
    
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


def _list_loki_label_values_impl(datasource_uid: str, label: str, start: Optional[str] = None,
                                end: Optional[str] = None, query: Optional[str] = None) -> pd.DataFrame:
    """
    List values for a specific log label from a Loki datasource.
    
    Args:
        datasource_uid (str): UID of the Loki datasource
        label (str): Label name to get values for
        start (Optional[str]): Start time (RFC3339 format)
        end (Optional[str]): End time (RFC3339 format)
        query (Optional[str]): LogQL query to filter results
        
    Returns:
        pd.DataFrame: DataFrame with label values
    """
    session, grafana_url = _get_grafana_session()
    
    endpoint = f"/api/datasources/proxy/{datasource_uid}/loki/api/v1/label/{label}/values"
    url = urljoin(grafana_url, endpoint)
    
    params = {}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    if query:
        params["query"] = query
    
    logger.info(f"Fetching Loki label values for label '{label}' in datasource: {datasource_uid}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    result = response.json()
    label_values = result.get("data", [])
    logger.info(f"Retrieved {len(label_values)} values for label '{label}' from Loki")
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'label_name': [label] * len(label_values),
        'label_value': label_values,
        'datasource_uid': [datasource_uid] * len(label_values),
        'start_time': [start] * len(label_values) if start else [None] * len(label_values),
        'end_time': [end] * len(label_values) if end else [None] * len(label_values),
        'filter_query': [query] * len(label_values) if query else [None] * len(label_values)
    })
    
    return df

_SHELL.push({"list_loki_label_values_impl": _list_loki_label_values_impl})


@app.tool()
async def list_loki_label_values(datasource_uid: str, label: str, start: Optional[str] = None,
                                end: Optional[str] = None, query: Optional[str] = None,
                                *, save_as: str) -> Optional[pd.DataFrame]:
    """
    List values for a specific log label from a Loki datasource.
    
    Args:
        datasource_uid (str): UID of the Loki datasource
        label (str): Label name to get values for
        start (Optional[str]): Start time (RFC3339 format)
        end (Optional[str]): End time (RFC3339 format)
        query (Optional[str]): LogQL query to filter results
        save_as (str): Variable name to store the label values
        
    Returns:
        pd.DataFrame: DataFrame with label values
    """
    code = f'{save_as} = list_loki_label_values_impl("{datasource_uid}", "{label}"'
    if start:
        code += f', "{start}"'
        if end:
            code += f', "{end}"'
            if query:
                code += f', "{query}"'
    code += f')\n{save_as}'
    
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


def _query_loki_stats_impl(datasource_uid: str, query: str, start: Optional[str] = None,
                          end: Optional[str] = None) -> pd.DataFrame:
    """
    Get statistics about log streams from a Loki datasource.
    
    Args:
        datasource_uid (str): UID of the Loki datasource
        query (str): LogQL query to get stats for
        start (Optional[str]): Start time (RFC3339 format)
        end (Optional[str]): End time (RFC3339 format)
        
    Returns:
        pd.DataFrame: Statistics about log streams as a DataFrame
    """
    session, grafana_url = _get_grafana_session()
    
    endpoint = f"/api/datasources/proxy/{datasource_uid}/loki/api/v1/index/stats"
    url = urljoin(grafana_url, endpoint)
    
    params = {"query": query}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    
    logger.info(f"Fetching Loki stats for query: {query}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    result = response.json()
    logger.info(f"Retrieved Loki stats successfully")
    
    # Convert stats to DataFrame
    data = result.get('data', {})
    stats_data = {
        'query': [query],
        'datasource_uid': [datasource_uid],
        'start_time': [start] if start else [None],
        'end_time': [end] if end else [None]
    }
    
    # Add all stats fields to the DataFrame
    for key, value in data.items():
        stats_data[key] = [value]
    
    return pd.DataFrame(stats_data)

_SHELL.push({"query_loki_stats_impl": _query_loki_stats_impl})


@app.tool()
async def query_loki_stats(datasource_uid: str, query: str, start: Optional[str] = None,
                          end: Optional[str] = None, *, save_as: str) -> Optional[pd.DataFrame]:
    """
    Get statistics about log streams from a Loki datasource.
    
    Args:
        datasource_uid (str): UID of the Loki datasource
        query (str): LogQL query to get stats for
        start (Optional[str]): Start time (RFC3339 format)
        end (Optional[str]): End time (RFC3339 format)
        save_as (str): Variable name to store the statistics
        
    Returns:
        pd.DataFrame: Statistics about log streams as a DataFrame
    """
    code = f'{save_as} = query_loki_stats_impl("{datasource_uid}", "{query}"'
    if start:
        code += f', "{start}"'
        if end:
            code += f', "{end}"'
    code += f')\n{save_as}'
    
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records') 