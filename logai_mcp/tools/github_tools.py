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
    
    session = requests.Session()
    session.headers.update({
        "Authorization": f"token {settings.github_pat_token}",
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

_SHELL.push({"get_issue_impl": _get_issue_impl})


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