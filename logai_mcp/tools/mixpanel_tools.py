"""Mixpanel Tools for LogAI MCP Server

This module provides tools for interacting with Mixpanel for analytics data retrieval.
All operations are logged and can be accessed through audit endpoints.
"""

import json
import requests
from typing import Optional, List
from urllib.parse import urljoin
import pandas as pd
from datetime import datetime, timedelta
import base64

from logai_mcp.session import app, logger
from logai_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from logai_mcp.config import get_settings


def _mixpanel_credentials_available() -> bool:
    """Check if Mixpanel credentials are available."""
    try:
        settings = get_settings()
        return bool(settings.mixpanel_api_secret)
    except Exception:
        return False


def _get_mixpanel_session():
    """Create a requests session with Mixpanel authentication headers."""
    settings = get_settings()
    if not settings.mixpanel_api_secret:
        raise ValueError("MIXPANEL_API_SECRET must be set in environment variables")
    
    # Use mixpanel_host if provided, otherwise default to mixpanel.com
    mixpanel_host = getattr(settings, 'mixpanel_host', 'https://mixpanel.com')
    if not mixpanel_host.startswith('http'):
        mixpanel_host = f"https://{mixpanel_host}"
    
    session = requests.Session()
    # Mixpanel uses HTTP Basic Auth with API secret as username and empty password
    auth_string = f"{settings.mixpanel_api_secret}:"
    auth_header = base64.b64encode(auth_string.encode()).decode()
    session.headers.update({
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/json"
    })
    
    return session, mixpanel_host


def _export_events_impl(from_date: str, to_date: str, event: Optional[str] = None,
                       where: Optional[str] = None, bucket: Optional[str] = None) -> pd.DataFrame:
    """
    Export raw event data from Mixpanel.
    
    Args:
        from_date (str): Start date in YYYY-MM-DD format
        to_date (str): End date in YYYY-MM-DD format
        event (Optional[str]): Specific event name to filter by
        where (Optional[str]): Expression to filter events
        bucket (Optional[str]): Data bucket/region (for EU residency)
        
    Returns:
        pd.DataFrame: Event data as a DataFrame
    """
    session, mixpanel_host = _get_mixpanel_session()
    
    # Use EU endpoint if bucket is specified
    if bucket and bucket.lower() == 'eu':
        base_url = "https://eu.mixpanel.com"
    else:
        base_url = mixpanel_host
    
    url = urljoin(base_url, "/api/2.0/export/")
    
    params = {
        'from_date': from_date,
        'to_date': to_date
    }
    
    if event:
        params['event'] = event
    if where:
        params['where'] = where
    
    logger.info(f"Exporting Mixpanel events from {from_date} to {to_date}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    # Mixpanel export returns JSONL (one JSON object per line)
    events = []
    for line in response.text.strip().split('\n'):
        if line:
            try:
                event_data = json.loads(line)
                events.append(event_data)
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Retrieved {len(events)} events")
    
    if not events:
        return pd.DataFrame(columns=['event', 'distinct_id', 'time', 'properties'])
    
    # Convert events to DataFrame
    rows = []
    for event_data in events:
        row = {
            'event': event_data.get('event'),
            'distinct_id': event_data.get('properties', {}).get('distinct_id'),
            'time': pd.to_datetime(event_data.get('properties', {}).get('time'), unit='s') if event_data.get('properties', {}).get('time') else None,
            'insert_id': event_data.get('properties', {}).get('$insert_id'),
            'ip': event_data.get('properties', {}).get('ip'),
            'city': event_data.get('properties', {}).get('$city'),
            'region': event_data.get('properties', {}).get('$region'),
            'country_code': event_data.get('properties', {}).get('mp_country_code'),
            'os': event_data.get('properties', {}).get('$os'),
            'browser': event_data.get('properties', {}).get('$browser'),
            'device': event_data.get('properties', {}).get('$device'),
            'screen_height': event_data.get('properties', {}).get('$screen_height'),
            'screen_width': event_data.get('properties', {}).get('$screen_width'),
            'lib_version': event_data.get('properties', {}).get('$lib_version'),
            'properties': json.dumps(event_data.get('properties', {}))
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def _query_insights_impl(event: str, unit: str = "day", interval: int = 1,
                        type: str = "general", from_date: Optional[str] = None, 
                        to_date: Optional[str] = None) -> pd.DataFrame:
    """
    Query Mixpanel Insights for event analytics.
    
    Args:
        event (str): Event name to query
        unit (str): Time unit for grouping (minute, hour, day, week, month)
        interval (int): Number of units per data point
        type (str): Query type (general, unique, average)
        from_date (Optional[str]): Start date in YYYY-MM-DD format
        to_date (Optional[str]): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: Insights data as a DataFrame
    """
    session, mixpanel_host = _get_mixpanel_session()
    
    url = urljoin(mixpanel_host, "/api/2.0/events/")
    
    # Default to last 30 days if dates not provided
    if not from_date:
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not to_date:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    params = {
        'event': event,
        'unit': unit,
        'interval': interval,
        'type': type,
        'from_date': from_date,
        'to_date': to_date
    }
    
    logger.info(f"Querying Mixpanel Insights for event: {event}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    result = response.json()
    logger.info(f"Retrieved Insights data for event: {event}")
    
    # Convert insights data to DataFrame
    rows = []
    event_data = result.get('data', {}).get('values', {}).get(event, {})
    
    for date_str, value in event_data.items():
        row = {
            'date': pd.to_datetime(date_str),
            'event': event,
            'value': value,
            'unit': unit,
            'interval': interval,
            'type': type
        }
        rows.append(row)
    
    if not rows:
        return pd.DataFrame(columns=['date', 'event', 'value', 'unit', 'interval', 'type'])
    
    return pd.DataFrame(rows).sort_values('date')


def _query_funnels_impl(events: List[str], unit: str = "day", interval: int = 1,
                       from_date: Optional[str] = None, to_date: Optional[str] = None,
                       on: Optional[str] = None) -> pd.DataFrame:
    """
    Query Mixpanel Funnels for conversion analytics.
    
    Args:
        events (List[str]): List of event names in funnel order
        unit (str): Time unit for the funnel window
        interval (int): Number of units for the funnel window
        from_date (Optional[str]): Start date in YYYY-MM-DD format
        to_date (Optional[str]): End date in YYYY-MM-DD format
        on (Optional[str]): Property to segment funnel by
        
    Returns:
        pd.DataFrame: Funnel data as a DataFrame
    """
    session, mixpanel_host = _get_mixpanel_session()
    
    url = urljoin(mixpanel_host, "/api/2.0/funnels/")
    
    # Default to last 30 days if dates not provided
    if not from_date:
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not to_date:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    params = {
        'events': json.dumps(events),
        'unit': unit,
        'interval': interval,
        'from_date': from_date,
        'to_date': to_date
    }
    
    if on:
        params['on'] = on
    
    logger.info(f"Querying Mixpanel Funnels for events: {events}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    result = response.json()
    logger.info(f"Retrieved Funnel data for {len(events)} events")
    
    # Convert funnel data to DataFrame
    rows = []
    funnel_data = result.get('data', {})
    
    for step_idx, step_data in enumerate(funnel_data):
        event_name = events[step_idx] if step_idx < len(events) else f"Step {step_idx + 1}"
        
        row = {
            'step': step_idx + 1,
            'event': event_name,
            'count': step_data.get('count', 0),
            'overall_conv_ratio': step_data.get('overall_conv_ratio', 0),
            'step_conv_ratio': step_data.get('step_conv_ratio', 0),
            'avg_time_to_convert': step_data.get('avg_time_to_convert'),
            'from_date': from_date,
            'to_date': to_date
        }
        rows.append(row)
    
    if not rows:
        return pd.DataFrame(columns=['step', 'event', 'count', 'overall_conv_ratio', 'step_conv_ratio', 'avg_time_to_convert', 'from_date', 'to_date'])
    
    return pd.DataFrame(rows)


def _query_retention_impl(retention_type: str = "birth", born_event: Optional[str] = None,
                         event: Optional[str] = None, born_where: Optional[str] = None,
                         where: Optional[str] = None, from_date: Optional[str] = None,
                         to_date: Optional[str] = None, unit: str = "day") -> pd.DataFrame:
    """
    Query Mixpanel Retention for user retention analytics.
    
    Args:
        retention_type (str): Type of retention analysis (birth, compounded)
        born_event (Optional[str]): Event that defines the birth cohort
        event (Optional[str]): Event to measure retention on
        born_where (Optional[str]): Filter for the birth event
        where (Optional[str]): Filter for the retention event
        from_date (Optional[str]): Start date in YYYY-MM-DD format
        to_date (Optional[str]): End date in YYYY-MM-DD format
        unit (str): Time unit for retention periods
        
    Returns:
        pd.DataFrame: Retention data as a DataFrame
    """
    session, mixpanel_host = _get_mixpanel_session()
    
    url = urljoin(mixpanel_host, "/api/2.0/retention/")
    
    # Default to last 30 days if dates not provided
    if not from_date:
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    if not to_date:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    params = {
        'retention_type': retention_type,
        'from_date': from_date,
        'to_date': to_date,
        'unit': unit
    }
    
    if born_event:
        params['born_event'] = born_event
    if event:
        params['event'] = event
    if born_where:
        params['born_where'] = born_where
    if where:
        params['where'] = where
    
    logger.info(f"Querying Mixpanel Retention from {from_date} to {to_date}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    result = response.json()
    logger.info(f"Retrieved Retention data")
    
    # Convert retention data to DataFrame
    rows = []
    data = result.get('data', [])
    
    for period_idx, period_data in enumerate(data):
        row = {
            'period': period_idx,
            'cohort_size': period_data.get('count', 0),
            'retention_rate': period_data.get('rate', 0),
            'retention_type': retention_type,
            'unit': unit,
            'from_date': from_date,
            'to_date': to_date
        }
        rows.append(row)
    
    if not rows:
        return pd.DataFrame(columns=['period', 'cohort_size', 'retention_rate', 'retention_type', 'unit', 'from_date', 'to_date'])
    
    return pd.DataFrame(rows)


def _query_people_impl(where: Optional[str] = None, selector: Optional[str] = None,
                      session_id: Optional[str] = None) -> pd.DataFrame:
    """
    Query Mixpanel People profiles.
    
    Args:
        where (Optional[str]): Expression to filter people profiles
        selector (Optional[str]): Specific properties to return
        session_id (Optional[str]): Session ID for paginated results
        
    Returns:
        pd.DataFrame: People profiles data as a DataFrame
    """
    session, mixpanel_host = _get_mixpanel_session()
    
    url = urljoin(mixpanel_host, "/api/2.0/engage/")
    
    params = {}
    if where:
        params['where'] = where
    if selector:
        params['selector'] = selector
    if session_id:
        params['session_id'] = session_id
    
    logger.info(f"Querying Mixpanel People profiles")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    result = response.json()
    logger.info(f"Retrieved {len(result.get('results', []))} people profiles")
    
    # Convert people data to DataFrame
    rows = []
    for profile in result.get('results', []):
        row = {
            'distinct_id': profile.get('$distinct_id'),
            'properties': json.dumps(profile.get('$properties', {})),
            'created': pd.to_datetime(profile.get('$properties', {}).get('$created')) if profile.get('$properties', {}).get('$created') else None,
            'last_seen': pd.to_datetime(profile.get('$properties', {}).get('$last_seen')) if profile.get('$properties', {}).get('$last_seen') else None,
            'email': profile.get('$properties', {}).get('$email'),
            'name': profile.get('$properties', {}).get('$name'),
            'first_name': profile.get('$properties', {}).get('$first_name'),
            'last_name': profile.get('$properties', {}).get('$last_name'),
            'city': profile.get('$properties', {}).get('$city'),
            'region': profile.get('$properties', {}).get('$region'),
            'country_code': profile.get('$properties', {}).get('mp_country_code')
        }
        rows.append(row)
    
    if not rows:
        return pd.DataFrame(columns=['distinct_id', 'properties', 'created', 'last_seen', 'email', 'name', 'first_name', 'last_name', 'city', 'region', 'country_code'])
    
    return pd.DataFrame(rows)


def _list_event_names_impl(type: str = "general", limit: int = 255) -> pd.DataFrame:
    """
    List all event names in the project.
    
    Args:
        type (str): Type of events to list (general, unique, average)
        limit (int): Maximum number of events to return
        
    Returns:
        pd.DataFrame: Event names as a DataFrame
    """
    session, mixpanel_host = _get_mixpanel_session()
    
    url = urljoin(mixpanel_host, "/api/2.0/events/names/")
    
    params = {
        'type': type,
        'limit': limit
    }
    
    logger.info(f"Listing Mixpanel event names")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    result = response.json()
    event_names = result if isinstance(result, list) else []
    logger.info(f"Retrieved {len(event_names)} event names")
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'event_name': event_names,
        'type': [type] * len(event_names)
    })
    
    return df


def _list_event_properties_impl(event: str, type: str = "general", limit: int = 255) -> pd.DataFrame:
    """
    List properties for a specific event.
    
    Args:
        event (str): Event name to get properties for
        type (str): Type of properties to list (general, unique, average)
        limit (int): Maximum number of properties to return
        
    Returns:
        pd.DataFrame: Event properties as a DataFrame
    """
    session, mixpanel_host = _get_mixpanel_session()
    
    url = urljoin(mixpanel_host, "/api/2.0/events/properties/")
    
    params = {
        'event': event,
        'type': type,
        'limit': limit
    }
    
    logger.info(f"Listing properties for event: {event}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    result = response.json()
    properties = result if isinstance(result, list) else []
    logger.info(f"Retrieved {len(properties)} properties for event: {event}")
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'event_name': [event] * len(properties),
        'property_name': properties,
        'type': [type] * len(properties)
    })
    
    return df


# Push implementation functions to shell
_SHELL.push({
    "export_events_impl": _export_events_impl,
    "query_insights_impl": _query_insights_impl,
    "query_funnels_impl": _query_funnels_impl,
    "query_retention_impl": _query_retention_impl,
    "query_people_impl": _query_people_impl,
    "list_event_names_impl": _list_event_names_impl,
    "list_event_properties_impl": _list_event_properties_impl,
})


# Conditional tool registration based on Mixpanel credentials
if _mixpanel_credentials_available():
    logger.info("Mixpanel credentials detected - registering Mixpanel tools")

    # MCP Tool implementations

    @app.tool()
    async def export_events(from_date: str, to_date: str, event: Optional[str] = None,
                           where: Optional[str] = None, bucket: Optional[str] = None, 
                           *, save_as: str) -> Optional[pd.DataFrame]:
        """
        Export raw event data from Mixpanel.
        
        Args:
            from_date (str): Start date in YYYY-MM-DD format
            to_date (str): End date in YYYY-MM-DD format
            event (Optional[str]): Specific event name to filter by
            where (Optional[str]): Expression to filter events
            bucket (Optional[str]): Data bucket/region (for EU residency)
            save_as (str): Variable name to store the event data
            
        Returns:
            pd.DataFrame: Event data as a DataFrame
        """
        code = f'{save_as} = export_events_impl("{from_date}", "{to_date}"'
        if event:
            code += f', "{event}"'
        else:
            code += ', None'
        if where:
            code += f', "{where}"'
        else:
            code += ', None'
        if bucket:
            code += f', "{bucket}"'
        code += f')\n{save_as}'
        
        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict('records')

    @app.tool()
    async def query_insights(event: str, unit: str = "day", interval: int = 1,
                            type: str = "general", from_date: Optional[str] = None,
                            to_date: Optional[str] = None, *, save_as: str) -> Optional[pd.DataFrame]:
        """
        Query Mixpanel Insights for event analytics.
        
        Args:
            event (str): Event name to query
            unit (str): Time unit for grouping (minute, hour, day, week, month)
            interval (int): Number of units per data point
            type (str): Query type (general, unique, average)
            from_date (Optional[str]): Start date in YYYY-MM-DD format
            to_date (Optional[str]): End date in YYYY-MM-DD format
            save_as (str): Variable name to store the insights data
            
        Returns:
            pd.DataFrame: Insights data as a DataFrame
        """
        code = f'{save_as} = query_insights_impl("{event}", "{unit}", {interval}, "{type}"'
        if from_date:
            code += f', "{from_date}"'
            if to_date:
                code += f', "{to_date}"'
        code += f')\n{save_as}'
        
        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict('records')

    @app.tool()
    async def query_funnels(events: str, unit: str = "day", interval: int = 1,
                           from_date: Optional[str] = None, to_date: Optional[str] = None,
                           on: Optional[str] = None, *, save_as: str) -> Optional[pd.DataFrame]:
        """
        Query Mixpanel Funnels for conversion analytics.
        
        Args:
            events (str): Comma-separated list of event names in funnel order
            unit (str): Time unit for the funnel window
            interval (int): Number of units for the funnel window
            from_date (Optional[str]): Start date in YYYY-MM-DD format
            to_date (Optional[str]): End date in YYYY-MM-DD format
            on (Optional[str]): Property to segment funnel by
            save_as (str): Variable name to store the funnel data
            
        Returns:
            pd.DataFrame: Funnel data as a DataFrame
        """
        events_list = [event.strip() for event in events.split(',')]
        
        code = f'{save_as} = query_funnels_impl({events_list}, "{unit}", {interval}'
        if from_date:
            code += f', "{from_date}"'
            if to_date:
                code += f', "{to_date}"'
                if on:
                    code += f', "{on}"'
        code += f')\n{save_as}'
        
        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict('records')

    @app.tool()
    async def query_retention(retention_type: str = "birth", born_event: Optional[str] = None,
                             event: Optional[str] = None, born_where: Optional[str] = None,
                             where: Optional[str] = None, from_date: Optional[str] = None,
                             to_date: Optional[str] = None, unit: str = "day",
                             *, save_as: str) -> Optional[pd.DataFrame]:
        """
        Query Mixpanel Retention for user retention analytics.
        
        Args:
            retention_type (str): Type of retention analysis (birth, compounded)
            born_event (Optional[str]): Event that defines the birth cohort
            event (Optional[str]): Event to measure retention on
            born_where (Optional[str]): Filter for the birth event
            where (Optional[str]): Filter for the retention event
            from_date (Optional[str]): Start date in YYYY-MM-DD format
            to_date (Optional[str]): End date in YYYY-MM-DD format
            unit (str): Time unit for retention periods
            save_as (str): Variable name to store the retention data
            
        Returns:
            pd.DataFrame: Retention data as a DataFrame
        """
        code = f'{save_as} = query_retention_impl("{retention_type}"'
        if born_event:
            code += f', "{born_event}"'
        else:
            code += ', None'
        if event:
            code += f', "{event}"'
        else:
            code += ', None'
        if born_where:
            code += f', "{born_where}"'
        else:
            code += ', None'
        if where:
            code += f', "{where}"'
        else:
            code += ', None'
        if from_date:
            code += f', "{from_date}"'
            if to_date:
                code += f', "{to_date}"'
        else:
            code += ', None, None'
        code += f', "{unit}")\n{save_as}'
        
        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict('records')

    @app.tool()
    async def query_people(where: Optional[str] = None, selector: Optional[str] = None,
                          session_id: Optional[str] = None, *, save_as: str) -> Optional[pd.DataFrame]:
        """
        Query Mixpanel People profiles.
        
        Args:
            where (Optional[str]): Expression to filter people profiles
            selector (Optional[str]): Specific properties to return
            session_id (Optional[str]): Session ID for paginated results
            save_as (str): Variable name to store the people data
            
        Returns:
            pd.DataFrame: People profiles data as a DataFrame
        """
        code = f'{save_as} = query_people_impl('
        if where:
            code += f'"{where}"'
        else:
            code += 'None'
        if selector:
            code += f', "{selector}"'
        else:
            code += ', None'
        if session_id:
            code += f', "{session_id}"'
        code += f')\n{save_as}'
        
        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict('records')

    @app.tool()
    async def list_event_names(type: str = "general", limit: int = 255, 
                              *, save_as: str) -> Optional[pd.DataFrame]:
        """
        List all event names in the project.
        
        Args:
            type (str): Type of events to list (general, unique, average)
            limit (int): Maximum number of events to return
            save_as (str): Variable name to store the event names
            
        Returns:
            pd.DataFrame: Event names as a DataFrame
        """
        code = f'{save_as} = list_event_names_impl("{type}", {limit})\n{save_as}'
        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict('records')

    @app.tool()
    async def list_event_properties(event: str, type: str = "general", limit: int = 255,
                                   *, save_as: str) -> Optional[pd.DataFrame]:
        """
        List properties for a specific event.
        
        Args:
            event (str): Event name to get properties for
            type (str): Type of properties to list (general, unique, average)
            limit (int): Maximum number of properties to return
            save_as (str): Variable name to store the event properties
            
        Returns:
            pd.DataFrame: Event properties as a DataFrame
        """
        code = f'{save_as} = list_event_properties_impl("{event}", "{type}", {limit})\n{save_as}'
        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict('records')

else:
    logger.info("Mixpanel credentials not detected - Mixpanel tools will not be registered") 