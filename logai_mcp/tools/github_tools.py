from typing import Union, Optional, Dict
import requests
import pandas as pd

from logai_mcp.session import (
    app,
    logger,
)

from logai_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from logai_mcp.config import get_settings


def _get_github_session():
    """Create a requests session with GitHub API authentication headers."""
    settings = get_settings()
    if not settings.github_pat_token:
        raise ValueError("GITHUB_PAT_TOKEN must be set in environment variables")
    
    # Validate token format
    token = settings.github_pat_token.strip()
    if not (token.startswith('ghp_') or token.startswith('github_pat_')):
        logger.warning(f"GitHub token format may be invalid. Expected to start with 'ghp_' or 'github_pat_', got: {token[:10]}...")
    
    session = requests.Session()
    session.headers.update({
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "LogAI-MCP-Server"
    })
    return session


# GitHub Issue Tools
def _get_issue_impl(owner: str, repo: str, issue_number: int) -> pd.DataFrame:
    """
    Get details of a specific issue from a GitHub repository.
    
    Args:
        owner (str): Repository owner (username or organization)
        repo (str): Repository name
        issue_number (int): Issue number to retrieve
        
    Returns:
        pd.DataFrame: Issue details as a DataFrame
    """
    session = _get_github_session()
    
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    
    logger.info(f"Fetching GitHub issue #{issue_number} from {owner}/{repo}")
    response = session.get(url)
    
    # Enhanced error handling with detailed information
    if not response.ok:
        error_details = {
            'status_code': response.status_code,
            'url': url,
            'headers': dict(response.headers),
            'response_text': response.text[:500] if response.text else "No response body"
        }
        
        if response.status_code == 403:
            rate_limit_remaining = response.headers.get('X-RateLimit-Remaining', 'Unknown')
            rate_limit_reset = response.headers.get('X-RateLimit-Reset', 'Unknown')
            error_msg = f"GitHub API 403 Forbidden: This could be due to:\n"
            error_msg += f"1. Invalid or insufficient token permissions\n"
            error_msg += f"2. Rate limiting (remaining: {rate_limit_remaining}, reset: {rate_limit_reset})\n"
            error_msg += f"3. Repository access restrictions\n"
            error_msg += f"4. Invalid repository or issue number\n"
            error_msg += f"URL: {url}\n"
            error_msg += f"Response: {response.text[:200]}"
            logger.error(error_msg)
            raise requests.exceptions.HTTPError(error_msg, response=response)
        elif response.status_code == 404:
            error_msg = f"GitHub API 404 Not Found: Repository '{owner}/{repo}' or issue #{issue_number} does not exist or is not accessible"
            logger.error(error_msg)
            raise requests.exceptions.HTTPError(error_msg, response=response)
        else:
            logger.error(f"GitHub API Error: {error_details}")
            response.raise_for_status()
    
    issue = response.json()
    logger.info(f"Retrieved issue: {issue['title']}")
    
    # Convert issue to DataFrame
    issue_data = {
        'number': [issue['number']],
        'title': [issue['title']],
        'body': [issue.get('body', '')],
        'state': [issue['state']],
        'user_login': [issue['user']['login']],
        'user_id': [issue['user']['id']],
        'created_at': [issue['created_at']],
        'updated_at': [issue['updated_at']],
        'closed_at': [issue.get('closed_at')],
        'labels': [', '.join([label['name'] for label in issue.get('labels', [])])],
        'assignees': [', '.join([assignee['login'] for assignee in issue.get('assignees', [])])],
        'milestone': [issue.get('milestone', {}).get('title') if issue.get('milestone') else None],
        'comments': [issue['comments']],
        'html_url': [issue['html_url']],
        'owner': [owner],
        'repo': [repo]
    }
    
    return pd.DataFrame(issue_data)


def _test_github_connection_impl() -> pd.DataFrame:
    """
    Test GitHub API connection and token validity.
    
    Returns:
        pd.DataFrame: Connection test results
    """
    try:
        session = _get_github_session()
        
        # Test 1: Check authenticated user
        logger.info("Testing GitHub API connection...")
        auth_response = session.get("https://api.github.com/user")
        
        # Test 2: Check rate limits
        rate_response = session.get("https://api.github.com/rate_limit")
        
        results = []
        
        # Auth test result
        if auth_response.ok:
            user_data = auth_response.json()
            results.append({
                'test': 'Authentication',
                'status': 'SUCCESS',
                'details': f"Logged in as: {user_data.get('login', 'Unknown')}",
                'user_id': user_data.get('id'),
                'user_type': user_data.get('type'),
                'scopes': auth_response.headers.get('X-OAuth-Scopes', 'Not available')
            })
        else:
            results.append({
                'test': 'Authentication', 
                'status': 'FAILED',
                'details': f"HTTP {auth_response.status_code}: {auth_response.text[:200]}",
                'user_id': None,
                'user_type': None,
                'scopes': None
            })
        
        # Rate limit test result  
        if rate_response.ok:
            rate_data = rate_response.json()
            core_limit = rate_data.get('resources', {}).get('core', {})
            results.append({
                'test': 'Rate Limits',
                'status': 'SUCCESS', 
                'details': f"Remaining: {core_limit.get('remaining', 'Unknown')}/{core_limit.get('limit', 'Unknown')}",
                'user_id': None,
                'user_type': None,
                'scopes': f"Reset at: {core_limit.get('reset', 'Unknown')}"
            })
        else:
            results.append({
                'test': 'Rate Limits',
                'status': 'FAILED',
                'details': f"HTTP {rate_response.status_code}: {rate_response.text[:200]}",
                'user_id': None,
                'user_type': None, 
                'scopes': None
            })
            
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.error(f"GitHub connection test failed: {e}")
        return pd.DataFrame([{
            'test': 'Connection',
            'status': 'FAILED', 
            'details': str(e),
            'user_id': None,
            'user_type': None,
            'scopes': None
        }])

_SHELL.push({"get_issue_impl": _get_issue_impl, "test_github_connection_impl": _test_github_connection_impl})


@app.tool()
async def get_issue(owner: str, repo: str, issue_number: int, *, save_as: str) -> Optional[pd.DataFrame]:
    """
    Get details of a specific issue from a GitHub repository.
    
    Args:
        owner (str): Repository owner (username or organization)
        repo (str): Repository name
        issue_number (int): Issue number to retrieve
        save_as (str): Variable name to store the issue details
        
    Returns:
        pd.DataFrame: Issue details as a DataFrame
    """
    code = f'{save_as} = get_issue_impl("{owner}", "{repo}", {issue_number})\n{save_as}'
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


@app.tool()
async def test_github_connection(*, save_as: str = "github_test_results") -> Optional[pd.DataFrame]:
    """
    Test GitHub API connection and token validity.
    
    Args:
        save_as (str): Variable name to store the test results
        
    Returns:
        pd.DataFrame: Test results showing connection status, user info, and rate limits
    """
    code = f'{save_as} = test_github_connection_impl()\n{save_as}'
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


def _search_issues_impl(owner: str, repo: str, query: Optional[str] = None, 
                       state: str = "open", labels: Optional[str] = None,
                       sort: str = "created", direction: str = "desc",
                       per_page: int = 30, page: int = 1) -> pd.DataFrame:
    """
    Search for issues in a GitHub repository.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        query (Optional[str]): Search query string
        state (str): Filter by state ('open', 'closed', 'all')
        labels (Optional[str]): Comma-separated list of labels
        sort (str): Sort by ('created', 'updated', 'comments')
        direction (str): Sort direction ('asc', 'desc')
        per_page (int): Results per page (max 100)
        page (int): Page number
        
    Returns:
        pd.DataFrame: Search results as a DataFrame
    """
    session = _get_github_session()
    
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    
    params = {
        'state': state,
        'sort': sort,
        'direction': direction,
        'per_page': min(per_page, 100),
        'page': page
    }
    
    if labels:
        params['labels'] = labels
    
    logger.info(f"Searching GitHub issues in {owner}/{repo} with state={state}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    issues = response.json()
    logger.info(f"Found {len(issues)} issues")
    
    if not issues:
        return pd.DataFrame(columns=['number', 'title', 'body', 'state', 'user_login', 'created_at', 'updated_at', 'owner', 'repo'])
    
    # Convert issues to DataFrame
    rows = []
    for issue in issues:
        # Skip pull requests (GitHub API returns PRs as issues)
        if 'pull_request' in issue:
            continue
            
        row = {
            'number': issue['number'],
            'title': issue['title'],
            'body': issue.get('body', ''),
            'state': issue['state'],
            'user_login': issue['user']['login'],
            'user_id': issue['user']['id'],
            'created_at': issue['created_at'],
            'updated_at': issue['updated_at'],
            'closed_at': issue.get('closed_at'),
            'labels': ', '.join([label['name'] for label in issue.get('labels', [])]),
            'assignees': ', '.join([assignee['login'] for assignee in issue.get('assignees', [])]),
            'milestone': issue.get('milestone', {}).get('title') if issue.get('milestone') else None,
            'comments': issue['comments'],
            'html_url': issue['html_url'],
            'owner': owner,
            'repo': repo
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

_SHELL.push({"search_issues_impl": _search_issues_impl})


@app.tool()
async def search_issues(owner: str, repo: str, query: Optional[str] = None,
                       state: str = "open", labels: Optional[str] = None,
                       sort: str = "created", direction: str = "desc",
                       per_page: int = 30, page: int = 1, *, save_as: str) -> Optional[pd.DataFrame]:
    """
    Search for issues in a GitHub repository.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        query (Optional[str]): Search query string
        state (str): Filter by state ('open', 'closed', 'all')
        labels (Optional[str]): Comma-separated list of labels
        sort (str): Sort by ('created', 'updated', 'comments')
        direction (str): Sort direction ('asc', 'desc')
        per_page (int): Results per page (max 100)
        page (int): Page number
        save_as (str): Variable name to store the search results
        
    Returns:
        pd.DataFrame: Search results as a DataFrame
    """
    code = f'{save_as} = search_issues_impl("{owner}", "{repo}"'
    if query:
        code += f', "{query}"'
    else:
        code += f', None'
    code += f', "{state}", '
    if labels:
        code += f'"{labels}"'
    else:
        code += f'None'
    code += f', "{sort}", "{direction}", {per_page}, {page})\n{save_as}'
    
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


# GitHub Pull Request Tools
def _get_pull_request_impl(owner: str, repo: str, pull_number: int) -> pd.DataFrame:
    """
    Get details of a specific pull request from a GitHub repository.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        pull_number (int): Pull request number
        
    Returns:
        pd.DataFrame: Pull request details as a DataFrame
    """
    session = _get_github_session()
    
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
    
    logger.info(f"Fetching GitHub pull request #{pull_number} from {owner}/{repo}")
    response = session.get(url)
    response.raise_for_status()
    
    pr = response.json()
    logger.info(f"Retrieved pull request: {pr['title']}")
    
    # Convert PR to DataFrame
    pr_data = {
        'number': [pr['number']],
        'title': [pr['title']],
        'body': [pr.get('body', '')],
        'state': [pr['state']],
        'user_login': [pr['user']['login']],
        'user_id': [pr['user']['id']],
        'created_at': [pr['created_at']],
        'updated_at': [pr['updated_at']],
        'closed_at': [pr.get('closed_at')],
        'merged_at': [pr.get('merged_at')],
        'merge_commit_sha': [pr.get('merge_commit_sha')],
        'head_ref': [pr['head']['ref']],
        'head_sha': [pr['head']['sha']],
        'base_ref': [pr['base']['ref']],
        'base_sha': [pr['base']['sha']],
        'draft': [pr.get('draft', False)],
        'merged': [pr.get('merged', False)],
        'mergeable': [pr.get('mergeable')],
        'mergeable_state': [pr.get('mergeable_state')],
        'comments': [pr['comments']],
        'review_comments': [pr['review_comments']],
        'commits': [pr['commits']],
        'additions': [pr['additions']],
        'deletions': [pr['deletions']],
        'changed_files': [pr['changed_files']],
        'labels': [', '.join([label['name'] for label in pr.get('labels', [])])],
        'assignees': [', '.join([assignee['login'] for assignee in pr.get('assignees', [])])],
        'requested_reviewers': [', '.join([reviewer['login'] for reviewer in pr.get('requested_reviewers', [])])],
        'html_url': [pr['html_url']],
        'diff_url': [pr['diff_url']],
        'patch_url': [pr['patch_url']],
        'owner': [owner],
        'repo': [repo]
    }
    
    return pd.DataFrame(pr_data)

_SHELL.push({"get_pull_request_impl": _get_pull_request_impl})


@app.tool()
async def get_pull_request(owner: str, repo: str, pull_number: int, *, save_as: str) -> Optional[pd.DataFrame]:
    """
    Get details of a specific pull request from a GitHub repository.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        pull_number (int): Pull request number
        save_as (str): Variable name to store the pull request details
        
    Returns:
        pd.DataFrame: Pull request details as a DataFrame
    """
    code = f'{save_as} = get_pull_request_impl("{owner}", "{repo}", {pull_number})\n{save_as}'
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


def _list_pull_requests_impl(owner: str, repo: str, state: str = "open",
                            head: Optional[str] = None, base: Optional[str] = None,
                            sort: str = "created", direction: str = "desc",
                            per_page: int = 30, page: int = 1) -> pd.DataFrame:
    """
    List pull requests in a GitHub repository.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        state (str): Filter by state ('open', 'closed', 'all')
        head (Optional[str]): Filter by head user/org and branch
        base (Optional[str]): Filter by base branch
        sort (str): Sort by ('created', 'updated', 'popularity', 'long-running')
        direction (str): Sort direction ('asc', 'desc')
        per_page (int): Results per page (max 100)
        page (int): Page number
        
    Returns:
        pd.DataFrame: Pull requests as a DataFrame
    """
    session = _get_github_session()
    
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    
    params = {
        'state': state,
        'sort': sort,
        'direction': direction,
        'per_page': min(per_page, 100),
        'page': page
    }
    
    if head:
        params['head'] = head
    if base:
        params['base'] = base
    
    logger.info(f"Listing GitHub pull requests in {owner}/{repo} with state={state}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    prs = response.json()
    logger.info(f"Found {len(prs)} pull requests")
    
    if not prs:
        return pd.DataFrame(columns=['number', 'title', 'body', 'state', 'user_login', 'created_at', 'updated_at', 'owner', 'repo'])
    
    # Convert PRs to DataFrame
    rows = []
    for pr in prs:
        row = {
            'number': pr['number'],
            'title': pr['title'],
            'body': pr.get('body', ''),
            'state': pr['state'],
            'user_login': pr['user']['login'],
            'user_id': pr['user']['id'],
            'created_at': pr['created_at'],
            'updated_at': pr['updated_at'],
            'closed_at': pr.get('closed_at'),
            'merged_at': pr.get('merged_at'),
            'head_ref': pr['head']['ref'],
            'head_sha': pr['head']['sha'],
            'base_ref': pr['base']['ref'],
            'base_sha': pr['base']['sha'],
            'draft': pr.get('draft', False),
            'merged': pr.get('merged', False),
            'labels': ', '.join([label['name'] for label in pr.get('labels', [])]),
            'assignees': ', '.join([assignee['login'] for assignee in pr.get('assignees', [])]),
            'requested_reviewers': ', '.join([reviewer['login'] for reviewer in pr.get('requested_reviewers', [])]),
            'html_url': pr['html_url'],
            'owner': owner,
            'repo': repo
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

_SHELL.push({"list_pull_requests_impl": _list_pull_requests_impl})


@app.tool()
async def list_pull_requests(owner: str, repo: str, state: str = "open",
                            head: Optional[str] = None, base: Optional[str] = None,
                            sort: str = "created", direction: str = "desc",
                            per_page: int = 30, page: int = 1, *, save_as: str) -> Optional[pd.DataFrame]:
    """
    List pull requests in a GitHub repository.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        state (str): Filter by state ('open', 'closed', 'all')
        head (Optional[str]): Filter by head user/org and branch
        base (Optional[str]): Filter by base branch
        sort (str): Sort by ('created', 'updated', 'popularity', 'long-running')
        direction (str): Sort direction ('asc', 'desc')
        per_page (int): Results per page (max 100)
        page (int): Page number
        save_as (str): Variable name to store the pull requests
        
    Returns:
        pd.DataFrame: Pull requests as a DataFrame
    """
    code = f'{save_as} = list_pull_requests_impl("{owner}", "{repo}", "{state}"'
    if head:
        code += f', "{head}"'
    else:
        code += f', None'
    if base:
        code += f', "{base}"'
    else:
        code += f', None'
    code += f', "{sort}", "{direction}", {per_page}, {page})\n{save_as}'
    
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


def _get_pull_request_files_impl(owner: str, repo: str, pull_number: int) -> pd.DataFrame:
    """
    Get the list of files changed in a pull request.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        pull_number (int): Pull request number
        
    Returns:
        pd.DataFrame: Changed files with patch and status details
    """
    session = _get_github_session()
    
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/files"
    
    logger.info(f"Fetching files for GitHub pull request #{pull_number} from {owner}/{repo}")
    response = session.get(url)
    response.raise_for_status()
    
    files = response.json()
    logger.info(f"Retrieved {len(files)} changed files")
    
    if not files:
        return pd.DataFrame(columns=['filename', 'status', 'additions', 'deletions', 'changes', 'owner', 'repo', 'pull_number'])
    
    # Convert files to DataFrame
    rows = []
    for file in files:
        row = {
            'filename': file['filename'],
            'status': file['status'],
            'additions': file['additions'],
            'deletions': file['deletions'],
            'changes': file['changes'],
            'blob_url': file.get('blob_url'),
            'raw_url': file.get('raw_url'),
            'contents_url': file.get('contents_url'),
            'patch': file.get('patch', ''),
            'previous_filename': file.get('previous_filename'),
            'sha': file.get('sha'),
            'owner': owner,
            'repo': repo,
            'pull_number': pull_number
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

_SHELL.push({"get_pull_request_files_impl": _get_pull_request_files_impl})


@app.tool()
async def get_pull_request_files(owner: str, repo: str, pull_number: int, *, save_as: str) -> Optional[pd.DataFrame]:
    """
    Get the list of files changed in a pull request.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        pull_number (int): Pull request number
        save_as (str): Variable name to store the file changes
        
    Returns:
        pd.DataFrame: Changed files with patch and status details
    """
    code = f'{save_as} = get_pull_request_files_impl("{owner}", "{repo}", {pull_number})\n{save_as}'
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


def _get_pull_request_comments_impl(owner: str, repo: str, pull_number: int) -> pd.DataFrame:
    """
    Get the review comments on a pull request.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        pull_number (int): Pull request number
        
    Returns:
        pd.DataFrame: Pull request review comments
    """
    session = _get_github_session()
    
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/comments"
    
    logger.info(f"Fetching comments for GitHub pull request #{pull_number} from {owner}/{repo}")
    response = session.get(url)
    response.raise_for_status()
    
    comments = response.json()
    logger.info(f"Retrieved {len(comments)} review comments")
    
    if not comments:
        return pd.DataFrame(columns=['id', 'body', 'user_login', 'created_at', 'updated_at', 'path', 'position', 'owner', 'repo', 'pull_number'])
    
    # Convert comments to DataFrame
    rows = []
    for comment in comments:
        row = {
            'id': comment['id'],
            'body': comment['body'],
            'user_login': comment['user']['login'],
            'user_id': comment['user']['id'],
            'created_at': comment['created_at'],
            'updated_at': comment['updated_at'],
            'path': comment.get('path'),
            'position': comment.get('position'),
            'original_position': comment.get('original_position'),
            'line': comment.get('line'),
            'original_line': comment.get('original_line'),
            'start_line': comment.get('start_line'),
            'original_start_line': comment.get('original_start_line'),
            'side': comment.get('side'),
            'start_side': comment.get('start_side'),
            'commit_id': comment.get('commit_id'),
            'original_commit_id': comment.get('original_commit_id'),
            'in_reply_to_id': comment.get('in_reply_to_id'),
            'html_url': comment['html_url'],
            'pull_request_review_id': comment.get('pull_request_review_id'),
            'owner': owner,
            'repo': repo,
            'pull_number': pull_number
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

_SHELL.push({"get_pull_request_comments_impl": _get_pull_request_comments_impl})


@app.tool()
async def get_pull_request_comments(owner: str, repo: str, pull_number: int, *, save_as: str) -> Optional[pd.DataFrame]:
    """
    Get the review comments on a pull request.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        pull_number (int): Pull request number
        save_as (str): Variable name to store the review comments
        
    Returns:
        pd.DataFrame: Pull request review comments
    """
    code = f'{save_as} = get_pull_request_comments_impl("{owner}", "{repo}", {pull_number})\n{save_as}'
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


def _get_pull_request_reviews_impl(owner: str, repo: str, pull_number: int) -> pd.DataFrame:
    """
    Get the reviews on a pull request.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        pull_number (int): Pull request number
        
    Returns:
        pd.DataFrame: Pull request reviews
    """
    session = _get_github_session()
    
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/reviews"
    
    logger.info(f"Fetching reviews for GitHub pull request #{pull_number} from {owner}/{repo}")
    response = session.get(url)
    response.raise_for_status()
    
    reviews = response.json()
    logger.info(f"Retrieved {len(reviews)} reviews")
    
    if not reviews:
        return pd.DataFrame(columns=['id', 'body', 'user_login', 'state', 'submitted_at', 'commit_id', 'owner', 'repo', 'pull_number'])
    
    # Convert reviews to DataFrame
    rows = []
    for review in reviews:
        row = {
            'id': review['id'],
            'body': review.get('body', ''),
            'user_login': review['user']['login'] if review['user'] else None,
            'user_id': review['user']['id'] if review['user'] else None,
            'state': review['state'],
            'submitted_at': review.get('submitted_at'),
            'commit_id': review['commit_id'],
            'html_url': review['html_url'],
            'pull_request_url': review['pull_request_url'],
            'author_association': review.get('author_association'),
            'owner': owner,
            'repo': repo,
            'pull_number': pull_number
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

_SHELL.push({"get_pull_request_reviews_impl": _get_pull_request_reviews_impl})


@app.tool()
async def get_pull_request_reviews(owner: str, repo: str, pull_number: int, *, save_as: str) -> Optional[pd.DataFrame]:
    """
    Get the reviews on a pull request.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        pull_number (int): Pull request number
        save_as (str): Variable name to store the reviews
        
    Returns:
        pd.DataFrame: Pull request reviews
    """
    code = f'{save_as} = get_pull_request_reviews_impl("{owner}", "{repo}", {pull_number})\n{save_as}'
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


def _list_commits_impl(owner: str, repo: str, sha: Optional[str] = None,
                      path: Optional[str] = None, author: Optional[str] = None,
                      since: Optional[str] = None, until: Optional[str] = None,
                      per_page: int = 30, page: int = 1) -> pd.DataFrame:
    """
    List commits in a GitHub repository.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        sha (Optional[str]): SHA or branch to start listing commits from
        path (Optional[str]): Only commits containing this file path
        author (Optional[str]): GitHub login or email address
        since (Optional[str]): Only commits after this date (ISO 8601)
        until (Optional[str]): Only commits before this date (ISO 8601)
        per_page (int): Results per page (max 100)
        page (int): Page number
        
    Returns:
        pd.DataFrame: Commits as a DataFrame
    """
    session = _get_github_session()
    
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    
    params: Dict[str, Union[str, int]] = {
        'per_page': min(per_page, 100),
        'page': page
    }
    
    if sha:
        params['sha'] = sha
    if path:
        params['path'] = path
    if author:
        params['author'] = author
    if since:
        params['since'] = since
    if until:
        params['until'] = until
    
    logger.info(f"Listing GitHub commits in {owner}/{repo}")
    response = session.get(url, params=params)
    response.raise_for_status()
    
    commits = response.json()
    logger.info(f"Retrieved {len(commits)} commits")
    
    if not commits:
        return pd.DataFrame(columns=['sha', 'message', 'author_name', 'author_email', 'committer_name', 'committer_email', 'date', 'owner', 'repo'])
    
    # Convert commits to DataFrame
    rows = []
    for commit in commits:
        commit_data = commit['commit']
        row = {
            'sha': commit['sha'],
            'message': commit_data['message'],
            'author_name': commit_data['author']['name'],
            'author_email': commit_data['author']['email'],
            'author_date': commit_data['author']['date'],
            'committer_name': commit_data['committer']['name'],
            'committer_email': commit_data['committer']['email'],
            'committer_date': commit_data['committer']['date'],
            'tree_sha': commit_data['tree']['sha'],
            'comment_count': commit_data['comment_count'],
            'verification_verified': commit_data.get('verification', {}).get('verified', False),
            'verification_reason': commit_data.get('verification', {}).get('reason'),
            'html_url': commit['html_url'],
            'comments_url': commit['comments_url'],
            'author_login': commit['author']['login'] if commit.get('author') else None,
            'committer_login': commit['committer']['login'] if commit.get('committer') else None,
            'parents_count': len(commit.get('parents', [])),
            'owner': owner,
            'repo': repo
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

_SHELL.push({"list_commits_impl": _list_commits_impl})


@app.tool()
async def list_commits(owner: str, repo: str, sha: Optional[str] = None,
                      path: Optional[str] = None, author: Optional[str] = None,
                      since: Optional[str] = None, until: Optional[str] = None,
                      per_page: int = 30, page: int = 1, *, save_as: str) -> Optional[pd.DataFrame]:
    """
    List commits in a GitHub repository.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        sha (Optional[str]): SHA or branch to start listing commits from
        path (Optional[str]): Only commits containing this file path
        author (Optional[str]): GitHub login or email address
        since (Optional[str]): Only commits after this date (ISO 8601)
        until (Optional[str]): Only commits before this date (ISO 8601)
        per_page (int): Results per page (max 100)
        page (int): Page number
        save_as (str): Variable name to store the commits
        
    Returns:
        pd.DataFrame: Commits as a DataFrame
    """
    code = f'{save_as} = list_commits_impl("{owner}", "{repo}"'
    
    # Handle optional parameters properly
    if sha is not None:
        code += f', "{sha}"'
    else:
        code += ', None'
        
    if path is not None:
        code += f', "{path}"'
    else:
        code += ', None'
        
    if author is not None:
        code += f', "{author}"'
    else:
        code += ', None'
        
    if since is not None:
        code += f', "{since}"'
    else:
        code += ', None'
        
    if until is not None:
        code += f', "{until}"'
    else:
        code += ', None'
        
    code += f', {per_page}, {page})\n{save_as}'
    
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


def _get_commit_details_impl(owner: str, repo: str, commit_sha: str) -> pd.DataFrame:
    """
    Get detailed information about a specific commit, including files changed and diffs.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        commit_sha (str): Commit SHA to get details for
        
    Returns:
        pd.DataFrame: Commit details including file changes and diffs
    """
    session = _get_github_session()
    
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"
    
    logger.info(f"Fetching GitHub commit details {commit_sha[:8]} from {owner}/{repo}")
    response = session.get(url)
    response.raise_for_status()
    
    commit = response.json()
    logger.info(f"Retrieved commit: {commit['commit']['message'][:50]}...")
    
    # Convert commit files to DataFrame rows
    rows = []
    files = commit.get('files', [])
    
    if not files:
        # If no files, create a single row with commit info
        row = {
            'sha': commit['sha'],
            'message': commit['commit']['message'],
            'author_name': commit['commit']['author']['name'],
            'author_email': commit['commit']['author']['email'],
            'author_date': commit['commit']['author']['date'],
            'committer_name': commit['commit']['committer']['name'],
            'committer_email': commit['commit']['committer']['email'],
            'committer_date': commit['commit']['committer']['date'],
            'total_additions': commit['stats']['additions'],
            'total_deletions': commit['stats']['deletions'],
            'total_changes': commit['stats']['total'],
            'files_changed_count': len(files),
            'filename': None,
            'file_status': None,
            'file_additions': None,
            'file_deletions': None,
            'file_changes': None,
            'file_patch': None,
            'html_url': commit['html_url'],
            'owner': owner,
            'repo': repo
        }
        rows.append(row)
    else:
        # Create a row for each file changed
        for file in files:
            row = {
                'sha': commit['sha'],
                'message': commit['commit']['message'],
                'author_name': commit['commit']['author']['name'],
                'author_email': commit['commit']['author']['email'],
                'author_date': commit['commit']['author']['date'],
                'committer_name': commit['commit']['committer']['name'],
                'committer_email': commit['commit']['committer']['email'],
                'committer_date': commit['commit']['committer']['date'],
                'total_additions': commit['stats']['additions'],
                'total_deletions': commit['stats']['deletions'],
                'total_changes': commit['stats']['total'],
                'files_changed_count': len(files),
                'filename': file['filename'],
                'file_status': file['status'],
                'file_additions': file['additions'],
                'file_deletions': file['deletions'],
                'file_changes': file['changes'],
                'file_patch': file.get('patch', ''),
                'file_blob_url': file.get('blob_url'),
                'file_raw_url': file.get('raw_url'),
                'previous_filename': file.get('previous_filename'),
                'html_url': commit['html_url'],
                'owner': owner,
                'repo': repo
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def _analyze_file_commits_around_issue_impl(owner: str, repo: str, issue_number: int, 
                                          file_paths: Optional[list] = None,
                                          days_before: int = 7, days_after: int = 1) -> pd.DataFrame:
    """
    Analyze commits to specific files around the time an issue was created.
    This helps identify what file changes might be responsible for an issue.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name  
        issue_number (int): Issue number to analyze
        file_paths (Optional[list]): Specific file paths to analyze (if None, gets all commits)
        days_before (int): Number of days before issue creation to look for commits
        days_after (int): Number of days after issue creation to look for commits
        
    Returns:
        pd.DataFrame: Commits around issue creation time with correlation analysis
    """
    from datetime import datetime, timedelta
    import pandas as pd
    
    session = _get_github_session()
    
    # First get the issue to find creation date
    issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    issue_response = session.get(issue_url)
    issue_response.raise_for_status()
    issue = issue_response.json()
    
    issue_created_at = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
    
    # Calculate time range
    since = (issue_created_at - timedelta(days=days_before)).isoformat()
    until = (issue_created_at + timedelta(days=days_after)).isoformat()
    
    logger.info(f"Analyzing commits around issue #{issue_number} created at {issue_created_at}")
    logger.info(f"Looking for commits between {since} and {until}")
    
    all_commits = []
    
    if file_paths:
        # Get commits for each specific file path
        for file_path in file_paths:
            commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
            params = {
                'path': file_path,
                'since': since,
                'until': until,
                'per_page': 100
            }
            
            logger.info(f"Getting commits for file: {file_path}")
            response = session.get(commits_url, params=params)
            response.raise_for_status()
            commits = response.json()
            
            for commit in commits:
                commit_data = commit['commit']
                commit_date = datetime.fromisoformat(commit_data['author']['date'].replace('Z', '+00:00'))
                
                # Calculate time difference from issue creation
                time_diff = (commit_date - issue_created_at).total_seconds() / 3600  # hours
                
                row = {
                    'issue_number': issue_number,
                    'issue_title': issue['title'],
                    'issue_created_at': issue['created_at'],
                    'issue_state': issue['state'],
                    'commit_sha': commit['sha'],
                    'commit_message': commit_data['message'],
                    'commit_author': commit_data['author']['name'],
                    'commit_date': commit_data['author']['date'],
                    'hours_from_issue': round(time_diff, 2),
                    'file_path': file_path,
                    'commit_url': commit['html_url'],
                    'owner': owner,
                    'repo': repo
                }
                all_commits.append(row)
    else:
        # Get all commits in the time range
        commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params = {
            'since': since,
            'until': until,
            'per_page': 100
        }
        
        logger.info("Getting all commits in time range")
        response = session.get(commits_url, params=params)
        response.raise_for_status()
        commits = response.json()
        
        for commit in commits:
            commit_data = commit['commit']
            commit_date = datetime.fromisoformat(commit_data['author']['date'].replace('Z', '+00:00'))
            
            # Calculate time difference from issue creation
            time_diff = (commit_date - issue_created_at).total_seconds() / 3600  # hours
            
            row = {
                'issue_number': issue_number,
                'issue_title': issue['title'],
                'issue_created_at': issue['created_at'],
                'issue_state': issue['state'],
                'commit_sha': commit['sha'],
                'commit_message': commit_data['message'],
                'commit_author': commit_data['author']['name'],
                'commit_date': commit_data['author']['date'],
                'hours_from_issue': round(time_diff, 2),
                'file_path': None,  # Will be filled if we fetch commit details
                'commit_url': commit['html_url'],
                'owner': owner,
                'repo': repo
            }
            all_commits.append(row)
    
    if not all_commits:
        logger.info("No commits found in the specified time range")
        return pd.DataFrame(columns=['issue_number', 'issue_title', 'commit_sha', 'commit_message', 'hours_from_issue'])
    
    df = pd.DataFrame(all_commits)
    # Sort by time difference (closest to issue creation first)
    df = df.sort_values('hours_from_issue', key=abs)
    
    logger.info(f"Found {len(df)} commits around issue #{issue_number}")
    return df

_SHELL.push({
    "get_commit_details_impl": _get_commit_details_impl,
    "analyze_file_commits_around_issue_impl": _analyze_file_commits_around_issue_impl
})


@app.tool()
async def get_commit_details(owner: str, repo: str, commit_sha: str, *, save_as: str) -> Optional[pd.DataFrame]:
    """
    Get detailed information about a specific commit, including files changed and diffs.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        commit_sha (str): Commit SHA to get details for
        save_as (str): Variable name to store the commit details
        
    Returns:
        pd.DataFrame: Commit details including file changes and diffs
    """
    code = f'{save_as} = get_commit_details_impl("{owner}", "{repo}", "{commit_sha}")\n{save_as}'
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records')


@app.tool()
async def analyze_file_commits_around_issue(owner: str, repo: str, issue_number: int,
                                          file_paths: Optional[str] = None,
                                          days_before: int = 7, days_after: int = 1, 
                                          *, save_as: str) -> Optional[pd.DataFrame]:
    """
    Analyze commits to specific files around the time an issue was created.
    This helps identify what file changes might be responsible for an issue.
    
    Args:
        owner (str): Repository owner
        repo (str): Repository name
        issue_number (int): Issue number to analyze
        file_paths (Optional[str]): Comma-separated list of file paths to analyze (if None, gets all commits)
        days_before (int): Number of days before issue creation to look for commits
        days_after (int): Number of days after issue creation to look for commits
        save_as (str): Variable name to store the analysis results
        
    Returns:
        pd.DataFrame: Commits around issue creation time with correlation analysis
    """
    file_paths_list = None
    if file_paths:
        file_paths_list = [path.strip() for path in file_paths.split(',')]
    
    code = f'{save_as} = analyze_file_commits_around_issue_impl("{owner}", "{repo}", {issue_number}'
    if file_paths_list:
        code += f', {file_paths_list}'
    else:
        code += ', None'
    code += f', {days_before}, {days_after})\n{save_as}'
    
    df = await run_code_in_shell(code)
    if isinstance(df, pd.DataFrame):
        return df.to_dict('records') 