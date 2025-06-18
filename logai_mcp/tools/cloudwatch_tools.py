"""AWS CloudWatch Tools for LogAI MCP Server

This module provides tools for interacting with Amazon CloudWatch for logs, metrics, and alarms.
All operations are logged and can be accessed through audit endpoints.
"""

import json
import boto3
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from botocore.exceptions import NoCredentialsError
from logai_mcp.session import app
from logai_mcp.config import get_settings
import logging

logger = logging.getLogger(__name__)


def _aws_credentials_available() -> bool:
    """Check if AWS credentials are available for CloudWatch."""
    try:
        # Try to create a session and get credentials
        session = boto3.Session()
        credentials = session.get_credentials()
        return credentials is not None
    except Exception:
        return False


def get_cloudwatch_logs_client():
    """Get configured CloudWatch Logs client with credentials from centralized config or AWS credential chain."""
    try:
        settings = get_settings()
        
        # Create session with credentials from settings if available
        session_kwargs = {}
        
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            session_kwargs['aws_access_key_id'] = settings.aws_access_key_id
            session_kwargs['aws_secret_access_key'] = settings.aws_secret_access_key
            
            if settings.aws_session_token:
                session_kwargs['aws_session_token'] = settings.aws_session_token
                
            logger.info("Using AWS credentials from configuration settings")
        else:
            # Fall back to default credential chain (environment variables, ~/.aws/credentials, IAM roles, etc.)
            logger.info("Using AWS default credential chain")
        
        session = boto3.Session(**session_kwargs)
        cloudwatch_logs_client = session.client('logs', region_name=settings.aws_region)
        
        # Test the connection by describing log groups (this will fail if no credentials)
        cloudwatch_logs_client.describe_log_groups(limit=1)
        
        logger.info(f"Successfully initialized CloudWatch Logs client for region: {settings.aws_region}")
        return cloudwatch_logs_client
        
    except NoCredentialsError:
        error_msg = """
        AWS credentials not found. Please configure credentials using one of these methods:
        
        1. Environment variables:
           export AWS_ACCESS_KEY_ID=your_access_key
           export AWS_SECRET_ACCESS_KEY=your_secret_key
           export AWS_REGION=us-east-1
        
        2. Configuration in .env file:
           AWS_ACCESS_KEY_ID=your_access_key
           AWS_SECRET_ACCESS_KEY=your_secret_key
           AWS_REGION=us-east-1
        
        3. AWS CLI: aws configure
        
        4. AWS credentials file (~/.aws/credentials)
        
        5. IAM roles (if running on EC2)
        """
        logger.error(error_msg)
        raise Exception(f"AWS credentials not configured: {error_msg}")
        
    except Exception as e:
        logger.error(f"Failed to initialize CloudWatch Logs client: {str(e)}")
        raise Exception(f"Failed to initialize CloudWatch Logs client: {str(e)}")


def get_cloudwatch_client():
    """Get configured CloudWatch client for metrics and alarms."""
    try:
        settings = get_settings()
        
        # Create session with credentials from settings if available
        session_kwargs = {}
        
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            session_kwargs['aws_access_key_id'] = settings.aws_access_key_id
            session_kwargs['aws_secret_access_key'] = settings.aws_secret_access_key
            
            if settings.aws_session_token:
                session_kwargs['aws_session_token'] = settings.aws_session_token
                
            logger.info("Using AWS credentials from configuration settings")
        else:
            logger.info("Using AWS default credential chain")
        
        session = boto3.Session(**session_kwargs)
        cloudwatch_client = session.client('cloudwatch', region_name=settings.aws_region)
        
        # Test the connection
        cloudwatch_client.list_metrics(MaxRecords=1)
        
        logger.info(f"Successfully initialized CloudWatch client for region: {settings.aws_region}")
        return cloudwatch_client
        
    except NoCredentialsError:
        error_msg = """
        AWS credentials not found. Please configure credentials using one of these methods:
        
        1. Environment variables:
           export AWS_ACCESS_KEY_ID=your_access_key
           export AWS_SECRET_ACCESS_KEY=your_secret_key
           export AWS_REGION=us-east-1
        
        2. Configuration in .env file:
           AWS_ACCESS_KEY_ID=your_access_key
           AWS_SECRET_ACCESS_KEY=your_secret_key
           AWS_REGION=us-east-1
        
        3. AWS CLI: aws configure
        
        4. AWS credentials file (~/.aws/credentials)
        
        5. IAM roles (if running on EC2)
        """
        logger.error(error_msg)
        raise Exception(f"AWS credentials not configured: {error_msg}")
        
    except Exception as e:
        logger.error(f"Failed to initialize CloudWatch client: {str(e)}")
        raise Exception(f"Failed to initialize CloudWatch client: {str(e)}")


def parse_time_string(time_str: str) -> datetime:
    """Parse time string in various formats to datetime object."""
    try:
        # Try ISO format first
        if 'T' in time_str:
            return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        
        # Try other common formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M',
            '%m/%d/%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Unable to parse time string: {time_str}")
        
    except Exception as e:
        raise ValueError(f"Invalid time format '{time_str}': {str(e)}")


# Conditional tool registration based on AWS credentials
if _aws_credentials_available():
    logger.info("AWS credentials detected - registering CloudWatch tools")

    # CloudWatch Logs Tools

    @app.tool()
    def list_log_groups(name_prefix: Optional[str] = None, limit: int = 50) -> str:
        """List CloudWatch log groups.
        
        Args:
            name_prefix: Filter log groups by name prefix (optional)
            limit: Maximum number of log groups to return (default 50, max 50)
        
        Returns:
            str: JSON string containing list of log groups with metadata
        """
        try:
            client = get_cloudwatch_logs_client()
            
            params: Dict[str, Any] = {'limit': min(limit, 50)}
            if name_prefix:
                params['logGroupNamePrefix'] = name_prefix
            
            response = client.describe_log_groups(**params)
            
            log_groups = []
            for log_group in response.get('logGroups', []):
                log_groups.append({
                    'name': log_group['logGroupName'],
                    'creation_time': datetime.fromtimestamp(log_group['creationTime'] / 1000).isoformat() if log_group.get('creationTime') else None,
                    'retention_in_days': log_group.get('retentionInDays'),
                    'stored_bytes': log_group.get('storedBytes', 0),
                    'metric_filter_count': log_group.get('metricFilterCount', 0),
                    'arn': log_group.get('arn')
                })
            
            result = {
                'log_groups': log_groups,
                'count': len(log_groups),
                'name_prefix': name_prefix
            }
            
            logger.info(f"Listed {len(log_groups)} CloudWatch log groups")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Failed to list CloudWatch log groups: {str(e)}"
            logger.error(error_msg)
            return json.dumps({'error': error_msg})


    @app.tool()
    def list_log_streams(log_group_name: str, name_prefix: Optional[str] = None, limit: int = 20) -> str:
        """List CloudWatch log streams in a log group.
        
        Args:
            log_group_name: Name of the log group
            name_prefix: Filter log streams by name prefix (optional)
            limit: Maximum number of log streams to return (default 20, max 50)
        
        Returns:
            str: JSON string containing list of log streams with metadata
        """
        try:
            client = get_cloudwatch_logs_client()
            
            params: Dict[str, Any] = {
                'logGroupName': log_group_name,
                'limit': min(limit, 50),
                'orderBy': 'LastEventTime',
                'descending': True
            }
            if name_prefix:
                params['logStreamNamePrefix'] = name_prefix
            
            response = client.describe_log_streams(**params)
            
            log_streams = []
            for stream in response.get('logStreams', []):
                log_streams.append({
                    'name': stream['logStreamName'],
                    'creation_time': datetime.fromtimestamp(stream['creationTime'] / 1000).isoformat() if stream.get('creationTime') else None,
                    'first_event_time': datetime.fromtimestamp(stream['firstEventTime'] / 1000).isoformat() if stream.get('firstEventTime') else None,
                    'last_event_time': datetime.fromtimestamp(stream['lastEventTime'] / 1000).isoformat() if stream.get('lastEventTime') else None,
                    'last_ingestion_time': datetime.fromtimestamp(stream['lastIngestionTime'] / 1000).isoformat() if stream.get('lastIngestionTime') else None,
                    'stored_bytes': stream.get('storedBytes', 0),
                    'arn': stream.get('arn')
                })
            
            result = {
                'log_group': log_group_name,
                'log_streams': log_streams,
                'count': len(log_streams),
                'name_prefix': name_prefix
            }
            
            logger.info(f"Listed {len(log_streams)} log streams in log group {log_group_name}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Failed to list log streams in {log_group_name}: {str(e)}"
            logger.error(error_msg)
            return json.dumps({'error': error_msg})


    @app.tool()
    def query_logs(log_group_names: List[str], query: str, start_time: str, end_time: Optional[str] = None, limit: int = 100) -> str:
        """Execute a CloudWatch Logs Insights query.
        
        Args:
            log_group_names: List of log group names to query
            query: CloudWatch Logs Insights query string
            start_time: Start time (ISO format, relative like '1h', '30m', '2d' ago, or absolute)
            end_time: End time (ISO format or relative) - defaults to now
            limit: Maximum number of log events to return (default 100, max 10000)
        
        Returns:
            str: JSON string containing query results
        """
        try:
            client = get_cloudwatch_logs_client()
            
            # Parse time inputs
            now = datetime.utcnow()
            
            # Handle relative time formats
            if start_time.endswith('m'):
                minutes = int(start_time[:-1])
                start_datetime = now - timedelta(minutes=minutes)
            elif start_time.endswith('h'):
                hours = int(start_time[:-1])
                start_datetime = now - timedelta(hours=hours)
            elif start_time.endswith('d'):
                days = int(start_time[:-1])
                start_datetime = now - timedelta(days=days)
            else:
                start_datetime = parse_time_string(start_time)
            
            if end_time:
                if end_time.endswith('m'):
                    minutes = int(end_time[:-1])
                    end_datetime = now - timedelta(minutes=minutes)
                elif end_time.endswith('h'):
                    hours = int(end_time[:-1])
                    end_datetime = now - timedelta(hours=hours)
                elif end_time.endswith('d'):
                    days = int(end_time[:-1])
                    end_datetime = now - timedelta(days=days)
                else:
                    end_datetime = parse_time_string(end_time)
            else:
                end_datetime = now
            
            # Convert to timestamps
            start_timestamp = int(start_datetime.timestamp())
            end_timestamp = int(end_datetime.timestamp())
            
            # Start the query
            response = client.start_query(
                logGroupNames=log_group_names,
                startTime=start_timestamp,
                endTime=end_timestamp,
                queryString=query,
                limit=min(limit, 10000)
            )
            
            query_id = response['queryId']
            
            # Poll for query completion
            import time
            max_wait = 60  # Maximum wait time in seconds
            wait_time = 0
            
            while wait_time < max_wait:
                result_response = client.get_query_results(queryId=query_id)
                status = result_response['status']
                
                if status == 'Complete':
                    break
                elif status == 'Failed':
                    error_msg = f"Query failed: {result_response.get('statistics', {}).get('recordsMatched', 'Unknown error')}"
                    logger.error(error_msg)
                    return json.dumps({'error': error_msg})
                
                time.sleep(2)
                wait_time += 2
            
            if wait_time >= max_wait:
                error_msg = "Query timed out after 60 seconds"
                logger.error(error_msg)
                return json.dumps({'error': error_msg})
            
            # Process results
            results = []
            for result in result_response.get('results', []):
                log_entry = {}
                for field in result:
                    log_entry[field['field']] = field['value']
                results.append(log_entry)
            
            query_result = {
                'query_id': query_id,
                'status': result_response['status'],
                'log_groups': log_group_names,
                'query': query,
                'start_time': start_datetime.isoformat(),
                'end_time': end_datetime.isoformat(),
                'results': results,
                'count': len(results),
                'statistics': result_response.get('statistics', {})
            }
            
            logger.info(f"Executed CloudWatch Logs query, returned {len(results)} results")
            return json.dumps(query_result, indent=2)
            
        except Exception as e:
            error_msg = f"Failed to execute CloudWatch Logs query: {str(e)}"
            logger.error(error_msg)
            return json.dumps({'error': error_msg})


    @app.tool()
    def get_log_events(log_group_name: str, log_stream_name: Optional[str] = None, 
                       start_time: Optional[str] = None, end_time: Optional[str] = None, 
                       filter_pattern: Optional[str] = None, limit: int = 100) -> str:
        """Get log events from CloudWatch Logs.
        
        Args:
            log_group_name: Name of the log group
            log_stream_name: Name of the log stream (optional, if not provided, searches all streams)
            start_time: Start time (ISO format or relative like '1h', '30m' ago)
            end_time: End time (ISO format or relative) - defaults to now
            filter_pattern: Filter pattern to match log events (optional)
            limit: Maximum number of events to return (default 100, max 10000)
        
        Returns:
            str: JSON string containing log events
        """
        try:
            client = get_cloudwatch_logs_client()
            
            # Parse time inputs
            now = datetime.utcnow()
            start_timestamp = None
            end_timestamp = None
            
            if start_time:
                if start_time.endswith('m'):
                    minutes = int(start_time[:-1])
                    start_datetime = now - timedelta(minutes=minutes)
                elif start_time.endswith('h'):
                    hours = int(start_time[:-1])
                    start_datetime = now - timedelta(hours=hours)
                elif start_time.endswith('d'):
                    days = int(start_time[:-1])
                    start_datetime = now - timedelta(days=days)
                else:
                    start_datetime = parse_time_string(start_time)
                start_timestamp = int(start_datetime.timestamp() * 1000)
            
            if end_time:
                if end_time.endswith('m'):
                    minutes = int(end_time[:-1])
                    end_datetime = now - timedelta(minutes=minutes)
                elif end_time.endswith('h'):
                    hours = int(end_time[:-1])
                    end_datetime = now - timedelta(hours=hours)
                elif end_time.endswith('d'):
                    days = int(end_time[:-1])
                    end_datetime = now - timedelta(days=days)
                else:
                    end_datetime = parse_time_string(end_time)
                end_timestamp = int(end_datetime.timestamp() * 1000)
            
            events = []
            
            if log_stream_name:
                # Get events from specific log stream
                params: Dict[str, Any] = {
                    'logGroupName': log_group_name,
                    'logStreamName': log_stream_name,
                    'limit': min(limit, 10000)
                }
                if start_timestamp:
                    params['startTime'] = start_timestamp
                if end_timestamp:
                    params['endTime'] = end_timestamp
                
                response = client.get_log_events(**params)
                
                for event in response.get('events', []):
                    events.append({
                        'timestamp': datetime.fromtimestamp(event['timestamp'] / 1000).isoformat(),
                        'message': event['message'],
                        'ingestion_time': datetime.fromtimestamp(event['ingestionTime'] / 1000).isoformat() if event.get('ingestionTime') else None,
                        'log_stream': log_stream_name
                    })
            else:
                # Search across all log streams in the log group
                params: Dict[str, Any] = {
                    'logGroupName': log_group_name,
                    'limit': min(limit, 10000)
                }
                if start_timestamp:
                    params['startTime'] = start_timestamp
                if end_timestamp:
                    params['endTime'] = end_timestamp
                if filter_pattern:
                    params['filterPattern'] = filter_pattern
                
                response = client.filter_log_events(**params)
                
                for event in response.get('events', []):
                    events.append({
                        'timestamp': datetime.fromtimestamp(event['timestamp'] / 1000).isoformat(),
                        'message': event['message'],
                        'ingestion_time': datetime.fromtimestamp(event['ingestionTime'] / 1000).isoformat() if event.get('ingestionTime') else None,
                        'log_stream': event.get('logStreamName'),
                        'event_id': event.get('eventId')
                    })
            
            result = {
                'log_group': log_group_name,
                'log_stream': log_stream_name,
                'filter_pattern': filter_pattern,
                'start_time': start_time,
                'end_time': end_time,
                'events': events,
                'count': len(events)
            }
            
            logger.info(f"Retrieved {len(events)} log events from {log_group_name}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Failed to get log events from {log_group_name}: {str(e)}"
            logger.error(error_msg)
            return json.dumps({'error': error_msg})


    # CloudWatch Metrics Tools

    @app.tool()
    def list_metrics(namespace: Optional[str] = None, metric_name: Optional[str] = None, 
                     dimensions: Optional[Dict[str, str]] = None, limit: int = 100) -> str:
        """List CloudWatch metrics.
        
        Args:
            namespace: AWS service namespace (e.g., AWS/EC2, AWS/Lambda)
            metric_name: Specific metric name to filter by
            dimensions: Dictionary of dimension name-value pairs to filter by
            limit: Maximum number of metrics to return (default 100, max 500)
        
        Returns:
            str: JSON string containing list of metrics
        """
        try:
            client = get_cloudwatch_client()
            
            params: Dict[str, Any] = {'MaxRecords': min(limit, 500)}
            
            if namespace:
                params['Namespace'] = namespace
            if metric_name:
                params['MetricName'] = metric_name
            if dimensions:
                params['Dimensions'] = [
                    {'Name': name, 'Value': value} 
                    for name, value in dimensions.items()
                ]
            
            response = client.list_metrics(**params)
            
            metrics = []
            for metric in response.get('Metrics', []):
                metric_info = {
                    'namespace': metric['Namespace'],
                    'metric_name': metric['MetricName'],
                    'dimensions': {dim['Name']: dim['Value'] for dim in metric.get('Dimensions', [])}
                }
                metrics.append(metric_info)
            
            result = {
                'metrics': metrics,
                'count': len(metrics),
                'filters': {
                    'namespace': namespace,
                    'metric_name': metric_name,
                    'dimensions': dimensions
                }
            }
            
            logger.info(f"Listed {len(metrics)} CloudWatch metrics")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Failed to list CloudWatch metrics: {str(e)}"
            logger.error(error_msg)
            return json.dumps({'error': error_msg})


    @app.tool()
    def get_metric_statistics(namespace: str, metric_name: str, start_time: str, end_time: Optional[str] = None,
                             period: int = 300, statistics: Optional[List[str]] = None, 
                             dimensions: Optional[Dict[str, str]] = None, unit: Optional[str] = None) -> str:
        """Get CloudWatch metric statistics.
        
        Args:
            namespace: AWS service namespace (e.g., AWS/EC2, AWS/Lambda)
            metric_name: Name of the metric
            start_time: Start time (ISO format or relative like '1h', '30m' ago)
            end_time: End time (ISO format or relative) - defaults to now
            period: Period in seconds for data points (minimum 60, default 300)
            statistics: List of statistics to retrieve (Average, Sum, Maximum, Minimum, SampleCount)
            dimensions: Dictionary of dimension name-value pairs
            unit: Unit of measurement for the metric
        
        Returns:
            str: JSON string containing metric statistics
        """
        try:
            client = get_cloudwatch_client()
            
            # Default statistics if none provided
            if not statistics:
                statistics = ['Average', 'Maximum', 'Minimum']
            
            # Parse time inputs
            now = datetime.utcnow()
            
            if start_time.endswith('m'):
                minutes = int(start_time[:-1])
                start_datetime = now - timedelta(minutes=minutes)
            elif start_time.endswith('h'):
                hours = int(start_time[:-1])
                start_datetime = now - timedelta(hours=hours)
            elif start_time.endswith('d'):
                days = int(start_time[:-1])
                start_datetime = now - timedelta(days=days)
            else:
                start_datetime = parse_time_string(start_time)
            
            if end_time:
                if end_time.endswith('m'):
                    minutes = int(end_time[:-1])
                    end_datetime = now - timedelta(minutes=minutes)
                elif end_time.endswith('h'):
                    hours = int(end_time[:-1])
                    end_datetime = now - timedelta(hours=hours)
                elif end_time.endswith('d'):
                    days = int(end_time[:-1])
                    end_datetime = now - timedelta(days=days)
                else:
                    end_datetime = parse_time_string(end_time)
            else:
                end_datetime = now
            
            params: Dict[str, Any] = {
                'Namespace': namespace,
                'MetricName': metric_name,
                'StartTime': start_datetime,
                'EndTime': end_datetime,
                'Period': max(period, 60),  # Minimum period is 60 seconds
                'Statistics': statistics
            }
            
            if dimensions:
                params['Dimensions'] = [
                    {'Name': name, 'Value': value}
                    for name, value in dimensions.items()
                ]
            
            if unit:
                params['Unit'] = unit
            
            response = client.get_metric_statistics(**params)
            
            # Sort datapoints by timestamp
            datapoints = sorted(response.get('Datapoints', []), key=lambda x: x['Timestamp'])
            
            # Format datapoints
            formatted_datapoints = []
            for point in datapoints:
                formatted_point = {
                    'timestamp': point['Timestamp'].isoformat(),
                    'unit': point.get('Unit')
                }
                for stat in statistics:
                    if stat in point:
                        formatted_point[stat.lower()] = point[stat]
                formatted_datapoints.append(formatted_point)
            
            result = {
                'namespace': namespace,
                'metric_name': metric_name,
                'dimensions': dimensions,
                'start_time': start_datetime.isoformat(),
                'end_time': end_datetime.isoformat(),
                'period_seconds': period,
                'statistics': statistics,
                'datapoints': formatted_datapoints,
                'count': len(formatted_datapoints),
                'label': response.get('Label')
            }
            
            logger.info(f"Retrieved {len(formatted_datapoints)} datapoints for metric {namespace}/{metric_name}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Failed to get metric statistics for {namespace}/{metric_name}: {str(e)}"
            logger.error(error_msg)
            return json.dumps({'error': error_msg})


    # CloudWatch Alarms Tools

    @app.tool()
    def list_alarms(alarm_names: Optional[List[str]] = None, alarm_name_prefix: Optional[str] = None,
                    state_value: Optional[str] = None, action_prefix: Optional[str] = None, 
                    limit: int = 100) -> str:
        """List CloudWatch alarms.
        
        Args:
            alarm_names: List of specific alarm names to retrieve
            alarm_name_prefix: Prefix to filter alarm names
            state_value: Filter by alarm state (OK, ALARM, INSUFFICIENT_DATA)
            action_prefix: Filter alarms that have actions starting with this prefix
            limit: Maximum number of alarms to return (default 100, max 100)
        
        Returns:
            str: JSON string containing list of alarms with their details
        """
        try:
            client = get_cloudwatch_client()
            
            params: Dict[str, Any] = {'MaxRecords': min(limit, 100)}
            
            if alarm_names:
                params['AlarmNames'] = alarm_names
            if alarm_name_prefix:
                params['AlarmNamePrefix'] = alarm_name_prefix
            if state_value:
                params['StateValue'] = state_value
            if action_prefix:
                params['ActionPrefix'] = action_prefix
            
            response = client.describe_alarms(**params)
            
            alarms = []
            for alarm in response.get('MetricAlarms', []):
                alarm_info = {
                    'name': alarm['AlarmName'],
                    'description': alarm.get('AlarmDescription'),
                    'state': alarm['StateValue'],
                    'state_reason': alarm.get('StateReason'),
                    'state_updated': alarm.get('StateUpdatedTimestamp').isoformat() if alarm.get('StateUpdatedTimestamp') else None,
                    'actions_enabled': alarm.get('ActionsEnabled'),
                    'ok_actions': alarm.get('OKActions', []),
                    'alarm_actions': alarm.get('AlarmActions', []),
                    'insufficient_data_actions': alarm.get('InsufficientDataActions', []),
                    'metric_name': alarm.get('MetricName'),
                    'namespace': alarm.get('Namespace'),
                    'statistic': alarm.get('Statistic'),
                    'dimensions': [{'name': dim['Name'], 'value': dim['Value']} for dim in alarm.get('Dimensions', [])],
                    'period': alarm.get('Period'),
                    'evaluation_periods': alarm.get('EvaluationPeriods'),
                    'threshold': alarm.get('Threshold'),
                    'comparison_operator': alarm.get('ComparisonOperator'),
                    'treat_missing_data': alarm.get('TreatMissingData'),
                    'alarm_arn': alarm.get('AlarmArn')
                }
                alarms.append(alarm_info)
            
            result = {
                'alarms': alarms,
                'count': len(alarms),
                'filters': {
                    'alarm_names': alarm_names,
                    'alarm_name_prefix': alarm_name_prefix,
                    'state_value': state_value,
                    'action_prefix': action_prefix
                }
            }
            
            logger.info(f"Listed {len(alarms)} CloudWatch alarms")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Failed to list CloudWatch alarms: {str(e)}"
            logger.error(error_msg)
            return json.dumps({'error': error_msg})


    @app.tool()
    def get_alarm_history(alarm_name: str, start_date: Optional[str] = None, end_date: Optional[str] = None,
                         history_item_type: Optional[str] = None, limit: int = 100) -> str:
        """Get CloudWatch alarm history.
        
        Args:
            alarm_name: Name of the alarm
            start_date: Start date for history (ISO format or relative like '1h', '30m' ago)
            end_date: End date for history (ISO format or relative) - defaults to now
            history_item_type: Type of history items (ConfigurationUpdate, StateUpdate, Action)
            limit: Maximum number of history items to return (default 100, max 100)
        
        Returns:
            str: JSON string containing alarm history
        """
        try:
            client = get_cloudwatch_client()
            
            params: Dict[str, Any] = {
                'AlarmName': alarm_name,
                'MaxRecords': min(limit, 100)
            }
            
            # Parse time inputs
            if start_date or end_date:
                now = datetime.utcnow()
                
                if start_date:
                    if start_date.endswith('m'):
                        minutes = int(start_date[:-1])
                        start_datetime = now - timedelta(minutes=minutes)
                    elif start_date.endswith('h'):
                        hours = int(start_date[:-1])
                        start_datetime = now - timedelta(hours=hours)
                    elif start_date.endswith('d'):
                        days = int(start_date[:-1])
                        start_datetime = now - timedelta(days=days)
                    else:
                        start_datetime = parse_time_string(start_date)
                    params['StartDate'] = start_datetime
                
                if end_date:
                    if end_date.endswith('m'):
                        minutes = int(end_date[:-1])
                        end_datetime = now - timedelta(minutes=minutes)
                    elif end_date.endswith('h'):
                        hours = int(end_date[:-1])
                        end_datetime = now - timedelta(hours=hours)
                    elif end_date.endswith('d'):
                        days = int(end_date[:-1])
                        end_datetime = now - timedelta(days=days)
                    else:
                        end_datetime = parse_time_string(end_date)
                    params['EndDate'] = end_datetime
            
            if history_item_type:
                params['HistoryItemType'] = history_item_type
            
            response = client.describe_alarm_history(**params)
            
            history_items = []
            for item in response.get('AlarmHistoryItems', []):
                history_item = {
                    'timestamp': item['Timestamp'].isoformat() if item.get('Timestamp') else None,
                    'history_item_type': item.get('HistoryItemType'),
                    'alarm_name': item.get('AlarmName'),
                    'history_data': item.get('HistoryData'),
                    'history_summary': item.get('HistorySummary')
                }
                history_items.append(history_item)
            
            result = {
                'alarm_name': alarm_name,
                'history_items': history_items,
                'count': len(history_items),
                'start_date': start_date,
                'end_date': end_date,
                'history_item_type': history_item_type
            }
            
            logger.info(f"Retrieved {len(history_items)} history items for alarm {alarm_name}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Failed to get alarm history for {alarm_name}: {str(e)}"
            logger.error(error_msg)
            return json.dumps({'error': error_msg})


    @app.tool()
    def put_metric_data(namespace: str, metric_data: List[Dict[str, Any]]) -> str:
        """Put custom metric data to CloudWatch.
        
        Args:
            namespace: Custom namespace for the metrics (should not start with AWS/)
            metric_data: List of metric data dictionaries, each containing:
                - MetricName (required): Name of the metric
                - Value (optional): Metric value (either Value or StatisticValues required)
                - Unit (optional): Unit of measurement
                - Timestamp (optional): ISO timestamp (defaults to current time)
                - Dimensions (optional): List of dimension dicts with Name and Value
                - StatisticValues (optional): Dict with Sum, SampleCount, Minimum, Maximum
        
        Returns:
            str: JSON string with put metric result
        """
        try:
            client = get_cloudwatch_client()
            
            # Validate and format metric data
            formatted_metrics = []
            for metric in metric_data:
                formatted_metric = {
                    'MetricName': metric['MetricName']
                }
                
                if 'Value' in metric:
                    formatted_metric['Value'] = float(metric['Value'])
                elif 'StatisticValues' in metric:
                    formatted_metric['StatisticValues'] = metric['StatisticValues']
                else:
                    raise ValueError(f"Metric {metric['MetricName']} must have either Value or StatisticValues")
                
                if 'Unit' in metric:
                    formatted_metric['Unit'] = metric['Unit']
                
                if 'Timestamp' in metric:
                    if isinstance(metric['Timestamp'], str):
                        formatted_metric['Timestamp'] = parse_time_string(metric['Timestamp'])
                    else:
                        formatted_metric['Timestamp'] = metric['Timestamp']
                else:
                    formatted_metric['Timestamp'] = datetime.utcnow()
                
                if 'Dimensions' in metric:
                    formatted_metric['Dimensions'] = metric['Dimensions']
                
                formatted_metrics.append(formatted_metric)
            
            # CloudWatch allows up to 20 metrics per request
            responses = []
            for i in range(0, len(formatted_metrics), 20):
                batch = formatted_metrics[i:i+20]
                response = client.put_metric_data(
                    Namespace=namespace,
                    MetricData=batch
                )
                responses.append(response)
            
            result = {
                'success': True,
                'namespace': namespace,
                'metrics_count': len(formatted_metrics),
                'batches_sent': len(responses),
                'message': f'Successfully put {len(formatted_metrics)} metrics to CloudWatch namespace {namespace}'
            }
            
            logger.info(f"Put {len(formatted_metrics)} metrics to CloudWatch namespace {namespace}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Failed to put metric data to namespace {namespace}: {str(e)}"
            logger.error(error_msg)
            return json.dumps({'error': error_msg})

else:
    logger.info("AWS credentials not detected - CloudWatch tools will not be registered") 