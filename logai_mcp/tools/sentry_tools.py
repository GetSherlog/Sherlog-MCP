"""Sentry Tools for LogAI MCP Server

This module provides tools for interacting with Sentry for error monitoring and issue tracking.
All operations are logged and can be accessed through audit endpoints.
"""

import json
import requests
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin
import pandas as pd

from logai_mcp.session import app, logger
from logai_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from logai_mcp.config import get_settings


def _sentry_credentials_available() -> bool:
    """Check if Sentry credentials are available."""
    try:
        settings = get_settings()
        return bool(settings.sentry_auth_token)
    except Exception:
        return False


def _get_sentry_session():
    """Create a requests session with Sentry authentication headers."""
    settings = get_settings()
    if not settings.sentry_auth_token:
        raise ValueError("SENTRY_AUTH_TOKEN must be set in environment variables")
    
    # Use sentry_host if provided, otherwise default to sentry.io
    sentry_host = getattr(settings, 'sentry_host', 'https://sentry.io')
    if not sentry_host.startswith('http'):
        sentry_host = f"https://{sentry_host}"
    
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {settings.sentry_auth_token}",
        "Content-Type": "application/json"
    })
    
    return session, sentry_host


def _list_projects_impl(organization_slug: str) -> pd.DataFrame:
    """
    List all accessible Sentry projects for a given organization.
    
    Args:
        organization_slug (str): The slug of the organization to list projects from
        
    Returns:
        pd.DataFrame: Projects data as a DataFrame
    """
    session, sentry_host = _get_sentry_session()
    
    url = urljoin(sentry_host, f"/api/0/organizations/{organization_slug}/projects/")
    
    logger.info(f"Fetching Sentry projects for organization: {organization_slug}")
    response = session.get(url)
    response.raise_for_status()
    
    projects = response.json()
    logger.info(f"Retrieved {len(projects)} projects")
    
    if not projects:
        return pd.DataFrame(columns=['id', 'name', 'slug', 'platform', 'status', 'organization_slug'])
    
    # Convert projects to DataFrame
    rows = []
    for project in projects:
        row = {
            'id': project['id'],
            'name': project['name'],
            'slug': project['slug'],
            'platform': project.get('platform'),
            'status': project.get('status'),
            'date_created': project.get('dateCreated'),
            'first_event': project.get('firstEvent'),
            'has_access': project.get('hasAccess', False),
            'is_member': project.get('isMember', False),
            'team_count': len(project.get('teams', [])),
            'color': project.get('color'),
            'organization_slug': organization_slug
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def _get_sentry_issue_impl(organization_slug: str, issue_id: str) -> pd.DataFrame:
    """
    Retrieve and analyze a Sentry issue by ID.
    
    Args:
        organization_slug (str): The slug of the organization
        issue_id (str): The issue ID to retrieve
        
    Returns:
        pd.DataFrame: Issue data as a DataFrame
    """
    session, sentry_host = _get_sentry_session()
    
    url = urljoin(sentry_host, f"/api/0/organizations/{organization_slug}/issues/{issue_id}/")
    
    logger.info(f"Fetching Sentry issue {issue_id} from organization: {organization_slug}")
    response = session.get(url)
    response.raise_for_status()
    
    issue = response.json()
    logger.info(f"Retrieved issue: {issue.get('title', 'No title')}")
    
    # Convert issue to DataFrame
    row = {
        'id': issue['id'],
        'short_id': issue.get('shortId'),
        'title': issue.get('title', ''),
        'culprit': issue.get('culprit', ''),
        'permalink': issue.get('permalink', ''),
        'logger': issue.get('logger'),
        'level': issue.get('level'),
        'status': issue.get('status'),
        'status_details': json.dumps(issue.get('statusDetails', {})),
        'is_public': issue.get('isPublic', False),
        'platform': issue.get('platform'),
        'project_id': issue.get('project', {}).get('id'),
        'project_name': issue.get('project', {}).get('name'),
        'project_slug': issue.get('project', {}).get('slug'),
        'type': issue.get('type'),
        'metadata': json.dumps(issue.get('metadata', {})),
        'num_comments': issue.get('numComments', 0),
        'user_count': issue.get('userCount', 0),
        'count': issue.get('count', 0),
        'user_report_count': issue.get('userReportCount', 0),
        'first_seen': issue.get('firstSeen'),
        'last_seen': issue.get('lastSeen'),
        'stats': json.dumps(issue.get('stats', {})),
        'annotations': json.dumps(issue.get('annotations', [])),
        'assigned_to': issue.get('assignedTo', {}).get('name') if issue.get('assignedTo') else None,
        'has_seen': issue.get('hasSeen', False),
        'is_subscribed': issue.get('isSubscribed', False),
        'is_bookmarked': issue.get('isBookmarked', False),
        'subscription_details': json.dumps(issue.get('subscriptionDetails', {})),
        'share_id': issue.get('shareId'),
        'organization_slug': organization_slug
    }
    
    return pd.DataFrame([row])


def _list_project_issues_impl(organization_slug: str, project_slug: str, 
                             query: Optional[str] = None, status: str = "unresolved",
                             sort: str = "date", limit: int = 25) -> pd.DataFrame:
    """
    List issues from a specific Sentry project.
    
    Args:
        organization_slug (str): The slug of the organization
        project_slug (str): The slug of the project
        query (Optional[str]): Search query to filter issues
        status (str): Issue status filter ('resolved', 'unresolved', 'ignored')
        sort (str): Sort order ('date', 'new', 'priority', 'freq', 'user')
        limit (int): Maximum number of issues to return
        
    Returns:
        pd.DataFrame: Issues data as a DataFrame
    """
    session, sentry_host = _get_sentry_session()
    
    url = urljoin(sentry_host, f"/api/0/projects/{organization_slug}/{project_slug}/issues/")
    
    params = {
        'statsPeriod': '24h',
        'query': f'is:{status}',
        'sort': sort,
        'limit': limit
    }
    
    if query:
        params['query'] += f' {query}'
    
    logger.info(f"Fetching Sentry issues for project: {organization_slug}/{project_slug}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    issues = response.json()
    logger.info(f"Retrieved {len(issues)} issues")
    
    if not issues:
        return pd.DataFrame(columns=['id', 'short_id', 'title', 'level', 'status', 'count', 'user_count', 'project_slug', 'organization_slug'])
    
    # Convert issues to DataFrame
    rows = []
    for issue in issues:
        row = {
            'id': issue['id'],
            'short_id': issue.get('shortId'),
            'title': issue.get('title', ''),
            'culprit': issue.get('culprit', ''),
            'level': issue.get('level'),
            'status': issue.get('status'),
            'platform': issue.get('platform'),
            'type': issue.get('type'),
            'count': issue.get('count', 0),
            'user_count': issue.get('userCount', 0),
            'first_seen': issue.get('firstSeen'),
            'last_seen': issue.get('lastSeen'),
            'permalink': issue.get('permalink', ''),
            'logger': issue.get('logger'),
            'is_public': issue.get('isPublic', False),
            'has_seen': issue.get('hasSeen', False),
            'is_subscribed': issue.get('isSubscribed', False),
            'is_bookmarked': issue.get('isBookmarked', False),
            'project_slug': project_slug,
            'organization_slug': organization_slug
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def _get_sentry_event_impl(organization_slug: str, issue_id: str, event_id: str) -> pd.DataFrame:
    """
    Retrieve and analyze a specific Sentry event from an issue.
    
    Args:
        organization_slug (str): The slug of the organization
        issue_id (str): The issue ID
        event_id (str): The specific event ID to retrieve
        
    Returns:
        pd.DataFrame: Event data as a DataFrame
    """
    session, sentry_host = _get_sentry_session()
    
    url = urljoin(sentry_host, f"/api/0/organizations/{organization_slug}/issues/{issue_id}/events/{event_id}/")
    
    logger.info(f"Fetching Sentry event {event_id} from issue {issue_id}")
    response = session.get(url)
    response.raise_for_status()
    
    event = response.json()
    logger.info(f"Retrieved event: {event.get('title', 'No title')}")
    
    # Convert event to DataFrame
    row = {
        'id': event['id'],
        'event_id': event.get('eventID'),
        'group_id': event.get('groupID'),
        'title': event.get('title', ''),
        'message': event.get('message', ''),
        'platform': event.get('platform'),
        'datetime': event.get('dateCreated'),
        'time_spent': event.get('timeSpent'),
        'tags': json.dumps(event.get('tags', [])),
        'user': json.dumps(event.get('user', {})),
        'contexts': json.dumps(event.get('contexts', {})),
        'packages': json.dumps(event.get('packages', {})),
        'sdk': json.dumps(event.get('sdk', {})),
        'errors': json.dumps(event.get('errors', [])),
        'fingerprint': json.dumps(event.get('fingerprint', [])),
        'stacktrace': json.dumps(event.get('exception', {}).get('values', []) if event.get('exception') else []),
        'release': event.get('release', {}).get('version') if event.get('release') else None,
        'environment': event.get('environment'),
        'size': event.get('size', 0),
        'dist': event.get('dist'),
        'organization_slug': organization_slug,
        'issue_id': issue_id
    }
    
    return pd.DataFrame([row])


def _list_issue_events_impl(organization_slug: str, issue_id: str, limit: int = 50) -> pd.DataFrame:
    """
    List events for a specific Sentry issue.
    
    Args:
        organization_slug (str): The slug of the organization
        issue_id (str): The ID of the issue to list events from
        limit (int): Maximum number of events to return
        
    Returns:
        pd.DataFrame: Events data as a DataFrame
    """
    session, sentry_host = _get_sentry_session()
    
    url = urljoin(sentry_host, f"/api/0/organizations/{organization_slug}/issues/{issue_id}/events/")
    
    params = {'limit': limit}
    
    logger.info(f"Fetching Sentry events for issue: {issue_id}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    events = response.json()
    logger.info(f"Retrieved {len(events)} events")
    
    if not events:
        return pd.DataFrame(columns=['id', 'event_id', 'title', 'message', 'platform', 'datetime', 'user', 'issue_id', 'organization_slug'])
    
    # Convert events to DataFrame
    rows = []
    for event in events:
        row = {
            'id': event['id'],
            'event_id': event.get('eventID'),
            'title': event.get('title', ''),
            'message': event.get('message', ''),
            'platform': event.get('platform'),
            'datetime': event.get('dateCreated'),
            'time_spent': event.get('timeSpent'),
            'user': json.dumps(event.get('user', {})),
            'tags': json.dumps(event.get('tags', [])),
            'size': event.get('size', 0),
            'release': event.get('release', {}).get('version') if event.get('release') else None,
            'environment': event.get('environment'),
            'dist': event.get('dist'),
            'issue_id': issue_id,
            'organization_slug': organization_slug
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def _resolve_short_id_impl(organization_slug: str, short_id: str) -> pd.DataFrame:
    """
    Retrieve details about an issue using its short ID.
    
    Args:
        organization_slug (str): The slug of the organization
        short_id (str): The short ID of the issue (e.g., PROJECT-123)
        
    Returns:
        pd.DataFrame: Issue data as a DataFrame
    """
    session, sentry_host = _get_sentry_session()
    
    url = urljoin(sentry_host, f"/api/0/organizations/{organization_slug}/shortids/{short_id}/")
    
    logger.info(f"Resolving Sentry short ID: {short_id}")
    response = session.get(url)
    response.raise_for_status()
    
    result = response.json()
    logger.info(f"Resolved short ID {short_id} to issue: {result.get('issue', {}).get('title', 'No title')}")
    
    # Extract issue data from result
    issue = result.get('issue', {})
    row = {
        'short_id': short_id,
        'issue_id': issue.get('id'),
        'title': issue.get('title', ''),
        'culprit': issue.get('culprit', ''),
        'permalink': issue.get('permalink', ''),
        'level': issue.get('level'),
        'status': issue.get('status'),
        'platform': issue.get('platform'),
        'project_id': issue.get('project', {}).get('id'),
        'project_name': issue.get('project', {}).get('name'),
        'project_slug': issue.get('project', {}).get('slug'),
        'count': issue.get('count', 0),
        'user_count': issue.get('userCount', 0),
        'first_seen': issue.get('firstSeen'),
        'last_seen': issue.get('lastSeen'),
        'organization_slug': organization_slug
    }
    
    return pd.DataFrame([row])


def _create_project_impl(organization_slug: str, team_slug: str, name: str, 
                        platform: Optional[str] = None) -> pd.DataFrame:
    """
    Create a new project in Sentry.
    
    Args:
        organization_slug (str): The slug of the organization
        team_slug (str): The slug of the team to assign the project to
        name (str): The name of the new project
        platform (Optional[str]): The platform for the new project
        
    Returns:
        pd.DataFrame: Created project data as a DataFrame
    """
    session, sentry_host = _get_sentry_session()
    
    url = urljoin(sentry_host, f"/api/0/teams/{organization_slug}/{team_slug}/projects/")
    
    payload = {
        'name': name,
        'platform': platform or 'other'
    }
    
    logger.info(f"Creating Sentry project: {name} for team: {team_slug}")
    response = session.post(url, json=payload)
    response.raise_for_status()
    
    project = response.json()
    logger.info(f"Created project: {project.get('name')} with ID: {project.get('id')}")
    
    # Convert project to DataFrame
    row = {
        'id': project['id'],
        'name': project['name'],
        'slug': project['slug'],
        'platform': project.get('platform'),
        'status': project.get('status'),
        'date_created': project.get('dateCreated'),
        'first_event': project.get('firstEvent'),
        'has_access': project.get('hasAccess', False),
        'is_member': project.get('isMember', False),
        'color': project.get('color'),
        'organization_slug': organization_slug,
        'team_slug': team_slug
    }
    
    return pd.DataFrame([row])


def _list_organization_replays_impl(organization_slug: str, project_ids: Optional[List[str]] = None,
                                   environment: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
    """
    List replays from a specific Sentry organization.
    
    Args:
        organization_slug (str): The slug of the organization
        project_ids (Optional[List[str]]): List of project IDs to filter by
        environment (Optional[str]): Environment to filter by
        limit (int): Maximum number of replays to return
        
    Returns:
        pd.DataFrame: Replays data as a DataFrame
    """
    session, sentry_host = _get_sentry_session()
    
    url = urljoin(sentry_host, f"/api/0/organizations/{organization_slug}/replays/")
    
    params: Dict[str, Any] = {'limit': limit}
    if project_ids:
        params['project'] = project_ids
    if environment:
        params['environment'] = environment
    
    logger.info(f"Fetching Sentry replays for organization: {organization_slug}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    replays = response.json()
    logger.info(f"Retrieved {len(replays)} replays")
    
    if not replays:
        return pd.DataFrame(columns=['id', 'replay_id', 'project_id', 'started_at', 'finished_at', 'duration', 'count_errors', 'organization_slug'])
    
    # Convert replays to DataFrame
    rows = []
    for replay in replays:
        row = {
            'id': replay.get('id'),
            'replay_id': replay.get('replayId'),
            'project_id': replay.get('projectId'),
            'started_at': replay.get('startedAt'),
            'finished_at': replay.get('finishedAt'),
            'duration': replay.get('duration'),
            'count_errors': replay.get('countErrors', 0),
            'count_segments': replay.get('countSegments', 0),
            'count_urls': replay.get('countUrls', 0),
            'user_id': replay.get('user', {}).get('id'),
            'user_username': replay.get('user', {}).get('username'),
            'user_email': replay.get('user', {}).get('email'),
            'browser_name': replay.get('browser', {}).get('name'),
            'browser_version': replay.get('browser', {}).get('version'),
            'device_name': replay.get('device', {}).get('name'),
            'device_brand': replay.get('device', {}).get('brand'),
            'os_name': replay.get('os', {}).get('name'),
            'os_version': replay.get('os', {}).get('version'),
            'organization_slug': organization_slug
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


# Push implementation functions to shell
_SHELL.push({
    "list_projects_impl": _list_projects_impl,
    "get_sentry_issue_impl": _get_sentry_issue_impl,
    "list_project_issues_impl": _list_project_issues_impl,
    "get_sentry_event_impl": _get_sentry_event_impl,
    "list_issue_events_impl": _list_issue_events_impl,
    "resolve_short_id_impl": _resolve_short_id_impl,
    "create_project_impl": _create_project_impl,
    "list_organization_replays_impl": _list_organization_replays_impl,
})


# Conditional tool registration based on Sentry credentials
if _sentry_credentials_available():
    logger.info("Sentry credentials detected - registering Sentry tools")

    # MCP Tool implementations

    @app.tool()
    async def list_projects(organization_slug: str, *, save_as: str) -> Optional[pd.DataFrame]:
        """
        List all accessible Sentry projects for a given organization.
        
        Args:
            organization_slug (str): The slug of the organization to list projects from
            save_as (str): Variable name to store the projects data
            
        Returns:
            pd.DataFrame: Projects data as a DataFrame
        """
        code = f'{save_as} = list_projects_impl("{organization_slug}")\n{save_as}'
        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict('records')

    @app.tool()
    async def get_sentry_issue(organization_slug: str, issue_id: str, *, save_as: str) -> Optional[pd.DataFrame]:
        """
        Retrieve and analyze a Sentry issue by ID.
        
        Args:
            organization_slug (str): The slug of the organization
            issue_id (str): The issue ID to retrieve
            save_as (str): Variable name to store the issue data
            
        Returns:
            pd.DataFrame: Issue data as a DataFrame
        """
        code = f'{save_as} = get_sentry_issue_impl("{organization_slug}", "{issue_id}")\n{save_as}'
        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict('records')

    @app.tool()
    async def list_project_issues(organization_slug: str, project_slug: str, 
                                 query: Optional[str] = None, status: str = "unresolved",
                                 sort: str = "date", limit: int = 25, *, save_as: str) -> Optional[pd.DataFrame]:
        """
        List issues from a specific Sentry project.
        
        Args:
            organization_slug (str): The slug of the organization
            project_slug (str): The slug of the project
            query (Optional[str]): Search query to filter issues
            status (str): Issue status filter ('resolved', 'unresolved', 'ignored')
            sort (str): Sort order ('date', 'new', 'priority', 'freq', 'user')
            limit (int): Maximum number of issues to return
            save_as (str): Variable name to store the issues data
            
        Returns:
            pd.DataFrame: Issues data as a DataFrame
        """
        code = f'{save_as} = list_project_issues_impl("{organization_slug}", "{project_slug}"'
        if query:
            code += f', "{query}"'
        else:
            code += ', None'
        code += f', "{status}", "{sort}", {limit})\n{save_as}'
        
        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict('records')

    @app.tool()
    async def get_sentry_event(organization_slug: str, issue_id: str, event_id: str, 
                              *, save_as: str) -> Optional[pd.DataFrame]:
        """
        Retrieve and analyze a specific Sentry event from an issue.
        
        Args:
            organization_slug (str): The slug of the organization
            issue_id (str): The issue ID
            event_id (str): The specific event ID to retrieve
            save_as (str): Variable name to store the event data
            
        Returns:
            pd.DataFrame: Event data as a DataFrame
        """
        code = f'{save_as} = get_sentry_event_impl("{organization_slug}", "{issue_id}", "{event_id}")\n{save_as}'
        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict('records')

    @app.tool()
    async def list_issue_events(organization_slug: str, issue_id: str, limit: int = 50, 
                               *, save_as: str) -> Optional[pd.DataFrame]:
        """
        List events for a specific Sentry issue.
        
        Args:
            organization_slug (str): The slug of the organization
            issue_id (str): The ID of the issue to list events from
            limit (int): Maximum number of events to return
            save_as (str): Variable name to store the events data
            
        Returns:
            pd.DataFrame: Events data as a DataFrame
        """
        code = f'{save_as} = list_issue_events_impl("{organization_slug}", "{issue_id}", {limit})\n{save_as}'
        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict('records')

    @app.tool()
    async def resolve_short_id(organization_slug: str, short_id: str, *, save_as: str) -> Optional[pd.DataFrame]:
        """
        Retrieve details about an issue using its short ID.
        
        Args:
            organization_slug (str): The slug of the organization
            short_id (str): The short ID of the issue (e.g., PROJECT-123)
            save_as (str): Variable name to store the issue data
            
        Returns:
            pd.DataFrame: Issue data as a DataFrame
        """
        code = f'{save_as} = resolve_short_id_impl("{organization_slug}", "{short_id}")\n{save_as}'
        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict('records')

    @app.tool()
    async def create_project(organization_slug: str, team_slug: str, name: str,
                            platform: Optional[str] = None, *, save_as: str) -> Optional[pd.DataFrame]:
        """
        Create a new project in Sentry.
        
        Args:
            organization_slug (str): The slug of the organization
            team_slug (str): The slug of the team to assign the project to
            name (str): The name of the new project
            platform (Optional[str]): The platform for the new project
            save_as (str): Variable name to store the created project data
            
        Returns:
            pd.DataFrame: Created project data as a DataFrame
        """
        code = f'{save_as} = create_project_impl("{organization_slug}", "{team_slug}", "{name}"'
        if platform:
            code += f', "{platform}"'
        code += f')\n{save_as}'
        
        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict('records')

    @app.tool()
    async def list_organization_replays(organization_slug: str, project_ids: Optional[str] = None,
                                       environment: Optional[str] = None, limit: int = 50,
                                       *, save_as: str) -> Optional[pd.DataFrame]:
        """
        List replays from a specific Sentry organization.
        
        Args:
            organization_slug (str): The slug of the organization
            project_ids (Optional[str]): Comma-separated list of project IDs to filter by
            environment (Optional[str]): Environment to filter by
            limit (int): Maximum number of replays to return
            save_as (str): Variable name to store the replays data
            
        Returns:
            pd.DataFrame: Replays data as a DataFrame
        """
        project_ids_list = None
        if project_ids:
            project_ids_list = [pid.strip() for pid in project_ids.split(',')]
        
        code = f'{save_as} = list_organization_replays_impl("{organization_slug}"'
        if project_ids_list:
            code += f', {project_ids_list}'
        else:
            code += ', None'
        if environment:
            code += f', "{environment}"'
        else:
            code += ', None'
        code += f', {limit})\n{save_as}'
        
        df = await run_code_in_shell(code)
        if isinstance(df, pd.DataFrame):
            return df.to_dict('records')

else:
    logger.info("Sentry credentials not detected - Sentry tools will not be registered") 