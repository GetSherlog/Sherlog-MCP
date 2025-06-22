"""GitHub Tools for Sherlog MCP Server

This module provides tools for interacting with GitHub repositories, issues, and pull requests.
All operations are logged and can be accessed through audit endpoints.

Tools are only registered if GitHub PAT token is available.
"""

import pandas as pd
import requests

from sherlog_mcp.config import get_settings
from sherlog_mcp.ipython_shell_utils import _SHELL, run_code_in_shell
from sherlog_mcp.session import (
    app,
    logger,
)

from datetime import timedelta
import dateutil.parser


def _github_credentials_available() -> bool:
    """Return True if GitHub PAT token is available."""
    try:
        settings = get_settings()
        return bool(settings.github_pat_token)
    except Exception:
        return False


if _github_credentials_available():
    logger.info("GitHub PAT token detected - registering GitHub tools")

    def _get_github_session():
        """Create a requests session with GitHub API authentication headers."""
        settings = get_settings()
        if not settings.github_pat_token:
            raise ValueError("GITHUB_PAT_TOKEN must be set in environment variables")

        token = settings.github_pat_token.strip()
        if not (token.startswith("ghp_") or token.startswith("github_pat_")):
            logger.warning(
                f"GitHub token format may be invalid. Expected to start with 'ghp_' or 'github_pat_', got: {token[:10]}..."
            )

        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "LogAI-MCP-Server",
            }
        )
        return session

    def _get_issue_impl(owner: str, repo: str, issue_number: int) -> pd.DataFrame:
        """Get details of a specific issue from a GitHub repository.

        Args:
            owner (str): Repository owner (username or organization)
            repo (str): Repository name
            issue_number (int): Issue number to retrieve

        Returns:
            pd.DataFrame: Issue details as a DataFrame

        """
        session = _get_github_session()

        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"

        response = session.get(url)

        if not response.ok:
            error_details = {
                "status_code": response.status_code,
                "url": url,
                "headers": dict(response.headers),
                "response_text": response.text[:500]
                if response.text
                else "No response body",
            }

            if response.status_code == 403:
                rate_limit_remaining = response.headers.get(
                    "X-RateLimit-Remaining", "Unknown"
                )
                rate_limit_reset = response.headers.get("X-RateLimit-Reset", "Unknown")
                error_msg = "GitHub API 403 Forbidden: This could be due to:\n"
                error_msg += "1. Invalid or insufficient token permissions\n"
                error_msg += f"2. Rate limiting (remaining: {rate_limit_remaining}, reset: {rate_limit_reset})\n"
                error_msg += "3. Repository access restrictions\n"
                error_msg += "4. Invalid repository or issue number\n"
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

        issue_data = {
            "number": [issue["number"]],
            "title": [issue["title"]],
            "body": [issue.get("body", "")],
            "state": [issue["state"]],
            "user_login": [issue["user"]["login"]],
            "user_id": [issue["user"]["id"]],
            "created_at": [issue["created_at"]],
            "updated_at": [issue["updated_at"]],
            "closed_at": [issue.get("closed_at")],
            "labels": [", ".join([label["name"] for label in issue.get("labels", [])])],
            "assignees": [
                ", ".join(
                    [assignee["login"] for assignee in issue.get("assignees", [])]
                )
            ],
            "milestone": [
                issue.get("milestone", {}).get("title")
                if issue.get("milestone")
                else None
            ],
            "comments": [issue["comments"]],
            "html_url": [issue["html_url"]],
            "owner": [owner],
            "repo": [repo],
        }

        return pd.DataFrame(issue_data)

    def _test_github_connection_impl() -> pd.DataFrame:
        """Test GitHub API connection and token validity.

        Returns:
            pd.DataFrame: Connection test results

        """
        try:
            session = _get_github_session()

            auth_response = session.get("https://api.github.com/user")
            rate_response = session.get("https://api.github.com/rate_limit")

            results = []

            if auth_response.ok:
                user_data = auth_response.json()
                results.append(
                    {
                        "test": "Authentication",
                        "status": "SUCCESS",
                        "details": f"Logged in as: {user_data.get('login', 'Unknown')}",
                        "user_id": user_data.get("id"),
                        "user_type": user_data.get("type"),
                        "scopes": auth_response.headers.get(
                            "X-OAuth-Scopes", "Not available"
                        ),
                    }
                )
            else:
                results.append(
                    {
                        "test": "Authentication",
                        "status": "FAILED",
                        "details": f"HTTP {auth_response.status_code}: {auth_response.text[:200]}",
                        "user_id": None,
                        "user_type": None,
                        "scopes": None,
                    }
                )

            if rate_response.ok:
                rate_data = rate_response.json()
                core_limit = rate_data.get("resources", {}).get("core", {})
                results.append(
                    {
                        "test": "Rate Limits",
                        "status": "SUCCESS",
                        "details": f"Remaining: {core_limit.get('remaining', 'Unknown')}/{core_limit.get('limit', 'Unknown')}",
                        "user_id": None,
                        "user_type": None,
                        "scopes": f"Reset at: {core_limit.get('reset', 'Unknown')}",
                    }
                )
            else:
                results.append(
                    {
                        "test": "Rate Limits",
                        "status": "FAILED",
                        "details": f"HTTP {rate_response.status_code}: {rate_response.text[:200]}",
                        "user_id": None,
                        "user_type": None,
                        "scopes": None,
                    }
                )

            return pd.DataFrame(results)

        except Exception as e:
            logger.error(f"GitHub connection test failed: {e}")
            return pd.DataFrame(
                [
                    {
                        "test": "Connection",
                        "status": "FAILED",
                        "details": str(e),
                        "user_id": None,
                        "user_type": None,
                        "scopes": None,
                    }
                ]
            )

    _SHELL.push(
        {
            "get_issue_impl": _get_issue_impl,
            "test_github_connection_impl": _test_github_connection_impl,
        }
    )

    @app.tool()
    async def get_issue(
        owner: str, repo: str, issue_number: int, *, save_as: str
    ) -> pd.DataFrame | None:
        """Get details of a specific issue from a GitHub repository.

        Args:
            owner (str): Repository owner (username or organization)
            repo (str): Repository name
            issue_number (int): Issue number to retrieve
            save_as (str): Variable name to store the issue details

        Returns:
            pd.DataFrame: Issue details as a DataFrame
            
        Examples
        --------
        After calling this tool with save_as="issue":
        
        # View the full issue details (single row DataFrame)
        >>> execute_python_code("issue")
        
        # View as a dictionary for easier reading
        >>> execute_python_code("issue.iloc[0].to_dict()")
        
        # Access specific fields
        >>> execute_python_code("issue.iloc[0]['title']")
        >>> execute_python_code("issue.iloc[0]['body']")
        >>> execute_python_code("issue.iloc[0]['state']")
        
        # View all column names
        >>> execute_python_code("issue.columns.tolist()")
        
        # Check labels and assignees
        >>> execute_python_code("issue.iloc[0][['labels', 'assignees']]")

        """
        code = f'{save_as} = get_issue_impl("{owner}", "{repo}", {issue_number})\n{save_as}'
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    @app.tool()
    async def test_github_connection(
        *, save_as: str = "github_test_results"
    ) -> pd.DataFrame | None:
        """Test GitHub API connection and token validity.

        Args:
            save_as (str): Variable name to store the test results

        Returns:
            pd.DataFrame: Test results showing connection status, user info, and rate limits

        Examples
        --------
        After calling this tool with save_as="github_test_results":
        
        # View all test results
        >>> execute_python_code("github_test_results")
        
        # Check if connection succeeded
        >>> execute_python_code("github_test_results[github_test_results['test'] == 'Connection']['status']")
        
        # View rate limit information
        >>> execute_python_code("github_test_results[github_test_results['test'] == 'Rate Limit']")
        
        # Get user information
        >>> execute_python_code("github_test_results[github_test_results['test'] == 'User Info']")

        """
        code = f"{save_as} = test_github_connection_impl()\n{save_as}"
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    def _search_issues_impl(
        owner: str,
        repo: str,
        query: str | None = None,
        state: str = "open",
        labels: str | None = None,
        sort: str = "created",
        direction: str = "desc",
        per_page: int = 30,
        page: int = 1,
    ) -> pd.DataFrame:
        """Search for issues in a GitHub repository.

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
            "state": state,
            "sort": sort,
            "direction": direction,
            "per_page": min(per_page, 100),
            "page": page,
        }

        if labels:
            params["labels"] = labels

        response = session.get(url, params=params)
        response.raise_for_status()

        issues = response.json()

        if not issues:
            return pd.DataFrame(
                columns=[
                    "number",
                    "title",
                    "body",
                    "state",
                    "user_login",
                    "created_at",
                    "updated_at",
                    "owner",
                    "repo",
                ]
            )

        rows = []
        for issue in issues:
            if "pull_request" in issue:
                continue

            row = {
                "number": issue["number"],
                "title": issue["title"],
                "body": issue.get("body", ""),
                "state": issue["state"],
                "user_login": issue["user"]["login"],
                "user_id": issue["user"]["id"],
                "created_at": issue["created_at"],
                "updated_at": issue["updated_at"],
                "closed_at": issue.get("closed_at"),
                "labels": ", ".join(
                    [label["name"] for label in issue.get("labels", [])]
                ),
                "assignees": ", ".join(
                    [assignee["login"] for assignee in issue.get("assignees", [])]
                ),
                "milestone": issue.get("milestone", {}).get("title")
                if issue.get("milestone")
                else None,
                "comments": issue["comments"],
                "html_url": issue["html_url"],
                "owner": owner,
                "repo": repo,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    _SHELL.push({"search_issues_impl": _search_issues_impl})

    @app.tool()
    async def search_issues(
        owner: str,
        repo: str,
        query: str | None = None,
        state: str = "open",
        labels: str | None = None,
        sort: str = "created",
        direction: str = "desc",
        per_page: int = 30,
        page: int = 1,
        *,
        save_as: str,
    ) -> pd.DataFrame | None:
        """Search for issues in a GitHub repository.

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
            
        Examples
        --------
        After calling this tool with save_as="issues":
        
        # View all issues
        >>> execute_python_code("issues")
        
        # Count issues by state
        >>> execute_python_code("issues['state'].value_counts()")
        
        # Filter for high priority issues (by labels)
        >>> execute_python_code("issues[issues['labels'].str.contains('high-priority')]")
        
        # View issues created in the last 7 days
        >>> execute_python_code("import pandas as pd; recent = issues[pd.to_datetime(issues['created_at']) > pd.Timestamp.now() - pd.Timedelta(days=7)]")
        
        # Sort by comment count
        >>> execute_python_code("issues.sort_values('comments', ascending=False).head(10)")
        
        # Get issue numbers and titles only
        >>> execute_python_code("issues[['number', 'title', 'state']]")
        
        # Export to CSV
        >>> execute_python_code("issues.to_csv('github_issues.csv', index=False)")

        """
        code = f'{save_as} = search_issues_impl("{owner}", "{repo}"'
        if query:
            code += f', "{query}"'
        else:
            code += ", None"
        code += f', "{state}", '
        if labels:
            code += f'"{labels}"'
        else:
            code += "None"
        code += f', "{sort}", "{direction}", {per_page}, {page})\n{save_as}'

        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    def _get_pull_request_impl(owner: str, repo: str, pull_number: int) -> pd.DataFrame:
        """Get details of a specific pull request from a GitHub repository.

        Args:
            owner (str): Repository owner
            repo (str): Repository name
            pull_number (int): Pull request number

        Returns:
            pd.DataFrame: Pull request details as a DataFrame

        """
        session = _get_github_session()

        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"

        response = session.get(url)
        response.raise_for_status()

        pr = response.json()

        pr_data = {
            "number": [pr["number"]],
            "title": [pr["title"]],
            "body": [pr.get("body", "")],
            "state": [pr["state"]],
            "user_login": [pr["user"]["login"]],
            "user_id": [pr["user"]["id"]],
            "created_at": [pr["created_at"]],
            "updated_at": [pr["updated_at"]],
            "closed_at": [pr.get("closed_at")],
            "merged_at": [pr.get("merged_at")],
            "merge_commit_sha": [pr.get("merge_commit_sha")],
            "head_ref": [pr["head"]["ref"]],
            "head_sha": [pr["head"]["sha"]],
            "base_ref": [pr["base"]["ref"]],
            "base_sha": [pr["base"]["sha"]],
            "draft": [pr.get("draft", False)],
            "merged": [pr.get("merged", False)],
            "mergeable": [pr.get("mergeable")],
            "mergeable_state": [pr.get("mergeable_state")],
            "comments": [pr["comments"]],
            "review_comments": [pr["review_comments"]],
            "commits": [pr["commits"]],
            "additions": [pr["additions"]],
            "deletions": [pr["deletions"]],
            "changed_files": [pr["changed_files"]],
            "labels": [", ".join([label["name"] for label in pr.get("labels", [])])],
            "assignees": [
                ", ".join([assignee["login"] for assignee in pr.get("assignees", [])])
            ],
            "requested_reviewers": [
                ", ".join(
                    [
                        reviewer["login"]
                        for reviewer in pr.get("requested_reviewers", [])
                    ]
                )
            ],
            "html_url": [pr["html_url"]],
            "diff_url": [pr["diff_url"]],
            "patch_url": [pr["patch_url"]],
            "owner": [owner],
            "repo": [repo],
        }

        return pd.DataFrame(pr_data)

    _SHELL.push({"get_pull_request_impl": _get_pull_request_impl})

    @app.tool()
    async def get_pull_request(
        owner: str, repo: str, pull_number: int, *, save_as: str
    ) -> pd.DataFrame | None:
        """Get details of a specific pull request from a GitHub repository.

        Args:
            owner (str): Repository owner
            repo (str): Repository name
            pull_number (int): Pull request number
            save_as (str): Variable name to store the pull request details

        Returns:
            pd.DataFrame: Pull request details as a DataFrame

        """
        code = f'{save_as} = get_pull_request_impl("{owner}", "{repo}", {pull_number})\n{save_as}'
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    def _list_pull_requests_impl(
        owner: str,
        repo: str,
        state: str = "open",
        head: str | None = None,
        base: str | None = None,
        sort: str = "created",
        direction: str = "desc",
        per_page: int = 30,
        page: int = 1,
    ) -> pd.DataFrame:
        """List pull requests in a GitHub repository.

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
            "state": state,
            "sort": sort,
            "direction": direction,
            "per_page": min(per_page, 100),
            "page": page,
        }

        if head:
            params["head"] = head
        if base:
            params["base"] = base

        response = session.get(url, params=params)
        response.raise_for_status()

        prs = response.json()

        if not prs:
            return pd.DataFrame(
                columns=[
                    "number",
                    "title",
                    "body",
                    "state",
                    "user_login",
                    "created_at",
                    "updated_at",
                    "owner",
                    "repo",
                ]
            )

        rows = []
        for pr in prs:
            row = {
                "number": pr["number"],
                "title": pr["title"],
                "body": pr.get("body", ""),
                "state": pr["state"],
                "user_login": pr["user"]["login"],
                "user_id": pr["user"]["id"],
                "created_at": pr["created_at"],
                "updated_at": pr["updated_at"],
                "closed_at": pr.get("closed_at"),
                "merged_at": pr.get("merged_at"),
                "head_ref": pr["head"]["ref"],
                "head_sha": pr["head"]["sha"],
                "base_ref": pr["base"]["ref"],
                "base_sha": pr["base"]["sha"],
                "draft": pr.get("draft", False),
                "merged": pr.get("merged", False),
                "labels": ", ".join([label["name"] for label in pr.get("labels", [])]),
                "assignees": ", ".join(
                    [assignee["login"] for assignee in pr.get("assignees", [])]
                ),
                "requested_reviewers": ", ".join(
                    [
                        reviewer["login"]
                        for reviewer in pr.get("requested_reviewers", [])
                    ]
                ),
                "html_url": pr["html_url"],
                "owner": owner,
                "repo": repo,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    _SHELL.push({"list_pull_requests_impl": _list_pull_requests_impl})

    @app.tool()
    async def list_pull_requests(
        owner: str,
        repo: str,
        state: str = "open",
        head: str | None = None,
        base: str | None = None,
        sort: str = "created",
        direction: str = "desc",
        per_page: int = 30,
        page: int = 1,
        *,
        save_as: str,
    ) -> pd.DataFrame | None:
        """List pull requests in a GitHub repository.

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
            save_as (str): Variable name to store the pull request list

        Returns:
            pd.DataFrame: Pull requests as a DataFrame

        """
        code = f'{save_as} = list_pull_requests_impl("{owner}", "{repo}", "{state}"'
        if head:
            code += f', "{head}"'
        else:
            code += ", None"
        if base:
            code += f', "{base}"'
        else:
            code += ", None"
        code += f', "{sort}", "{direction}", {per_page}, {page})\n{save_as}'

        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    def _get_pull_request_files_impl(
        owner: str, repo: str, pull_number: int
    ) -> pd.DataFrame:
        """Get files changed in a pull request.

        Args:
            owner (str): Repository owner
            repo (str): Repository name
            pull_number (int): Pull request number

        Returns:
            pd.DataFrame: Files changed in the pull request

        """
        session = _get_github_session()

        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/files"

        response = session.get(url)
        response.raise_for_status()

        files = response.json()

        if not files:
            return pd.DataFrame(
                columns=[
                    "filename",
                    "status",
                    "additions",
                    "deletions",
                    "changes",
                    "owner",
                    "repo",
                    "pull_number",
                ]
            )

        rows = []
        for file in files:
            row = {
                "filename": file["filename"],
                "status": file["status"],
                "additions": file["additions"],
                "deletions": file["deletions"],
                "changes": file["changes"],
                "blob_url": file.get("blob_url"),
                "raw_url": file.get("raw_url"),
                "patch": file.get("patch", ""),
                "owner": owner,
                "repo": repo,
                "pull_number": pull_number,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    _SHELL.push({"get_pull_request_files_impl": _get_pull_request_files_impl})

    @app.tool()
    async def get_pull_request_files(
        owner: str, repo: str, pull_number: int, *, save_as: str
    ) -> pd.DataFrame | None:
        """Get files changed in a pull request.

        Args:
            owner (str): Repository owner
            repo (str): Repository name
            pull_number (int): Pull request number
            save_as (str): Variable name to store the file list

        Returns:
            pd.DataFrame: Files changed in the pull request

        """
        code = f'{save_as} = get_pull_request_files_impl("{owner}", "{repo}", {pull_number})\n{save_as}'
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    def _get_pull_request_comments_impl(
        owner: str, repo: str, pull_number: int
    ) -> pd.DataFrame:
        """Get comments on a pull request.

        Args:
            owner (str): Repository owner
            repo (str): Repository name
            pull_number (int): Pull request number

        Returns:
            pd.DataFrame: Comments on the pull request

        """
        session = _get_github_session()

        url = (
            f"https://api.github.com/repos/{owner}/{repo}/issues/{pull_number}/comments"
        )

        response = session.get(url)
        response.raise_for_status()

        comments = response.json()

        if not comments:
            return pd.DataFrame(
                columns=[
                    "id",
                    "user_login",
                    "body",
                    "created_at",
                    "updated_at",
                    "owner",
                    "repo",
                    "pull_number",
                ]
            )

        rows = []
        for comment in comments:
            row = {
                "id": comment["id"],
                "user_login": comment["user"]["login"],
                "user_id": comment["user"]["id"],
                "body": comment["body"],
                "created_at": comment["created_at"],
                "updated_at": comment["updated_at"],
                "html_url": comment["html_url"],
                "owner": owner,
                "repo": repo,
                "pull_number": pull_number,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    _SHELL.push({"_get_pull_request_comments_impl": _get_pull_request_comments_impl})

    @app.tool()
    async def get_pull_request_comments(
        owner: str, repo: str, pull_number: int, *, save_as: str
    ) -> pd.DataFrame | None:
        """Get comments on a pull request.

        Args:
            owner (str): Repository owner
            repo (str): Repository name
            pull_number (int): Pull request number
            save_as (str): Variable name to store the comments

        Returns:
            pd.DataFrame: Comments on the pull request

        """
        code = f'{save_as} = get_pull_request_comments_impl("{owner}", "{repo}", {pull_number})\n{save_as}'
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    def _get_pull_request_reviews_impl(
        owner: str, repo: str, pull_number: int
    ) -> pd.DataFrame:
        """Get reviews on a pull request.

        Args:
            owner (str): Repository owner
            repo (str): Repository name
            pull_number (int): Pull request number

        Returns:
            pd.DataFrame: Reviews on the pull request

        """
        session = _get_github_session()

        url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}/reviews"

        response = session.get(url)
        response.raise_for_status()

        reviews = response.json()

        if not reviews:
            return pd.DataFrame(
                columns=[
                    "id",
                    "user_login",
                    "state",
                    "body",
                    "submitted_at",
                    "owner",
                    "repo",
                    "pull_number",
                ]
            )

        rows = []
        for review in reviews:
            row = {
                "id": review["id"],
                "user_login": review["user"]["login"] if review["user"] else None,
                "user_id": review["user"]["id"] if review["user"] else None,
                "state": review["state"],
                "body": review.get("body", ""),
                "submitted_at": review.get("submitted_at"),
                "html_url": review["html_url"],
                "owner": owner,
                "repo": repo,
                "pull_number": pull_number,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    _SHELL.push({"get_pull_request_reviews_impl": _get_pull_request_reviews_impl})

    @app.tool()
    async def get_pull_request_reviews(
        owner: str, repo: str, pull_number: int, *, save_as: str
    ) -> pd.DataFrame | None:
        """Get reviews on a pull request.

        Args:
            owner (str): Repository owner
            repo (str): Repository name
            pull_number (int): Pull request number
            save_as (str): Variable name to store the reviews

        Returns:
            pd.DataFrame: Reviews on the pull request

        """
        code = f'{save_as} = get_pull_request_reviews_impl("{owner}", "{repo}", {pull_number})\n{save_as}'
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    def _list_commits_impl(
        owner: str,
        repo: str,
        sha: str | None = None,
        path: str | None = None,
        author: str | None = None,
        since: str | None = None,
        until: str | None = None,
        per_page: int = 30,
        page: int = 1,
    ) -> pd.DataFrame:
        """List commits in a GitHub repository.

        Args:
            owner (str): Repository owner
            repo (str): Repository name
            sha (Optional[str]): SHA or branch to start listing commits from
            path (Optional[str]): Only commits containing this file path
            author (Optional[str]): GitHub login or email address
            since (Optional[str]): ISO 8601 date string
            until (Optional[str]): ISO 8601 date string
            per_page (int): Results per page (max 100)
            page (int): Page number

        Returns:
            pd.DataFrame: Commits as a DataFrame

        """
        session = _get_github_session()

        url = f"https://api.github.com/repos/{owner}/{repo}/commits"

        params: dict[str, str | int] = {"per_page": min(per_page, 100), "page": page}

        if sha:
            params["sha"] = sha
        if path:
            params["path"] = path
        if author:
            params["author"] = author
        if since:
            params["since"] = since
        if until:
            params["until"] = until

        response = session.get(url, params=params)
        response.raise_for_status()

        commits = response.json()

        if not commits:
            return pd.DataFrame(
                columns=[
                    "sha",
                    "message",
                    "author_name",
                    "author_email",
                    "author_date",
                    "committer_name",
                    "committer_email",
                    "committer_date",
                    "owner",
                    "repo",
                ]
            )

        rows = []
        for commit in commits:
            commit_data = commit["commit"]
            row = {
                "sha": commit["sha"],
                "message": commit_data["message"],
                "author_name": commit_data["author"]["name"],
                "author_email": commit_data["author"]["email"],
                "author_date": commit_data["author"]["date"],
                "committer_name": commit_data["committer"]["name"],
                "committer_email": commit_data["committer"]["email"],
                "committer_date": commit_data["committer"]["date"],
                "html_url": commit["html_url"],
                "owner": owner,
                "repo": repo,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    _SHELL.push({"list_commits_impl": _list_commits_impl})

    @app.tool()
    async def list_commits(
        owner: str,
        repo: str,
        sha: str | None = None,
        path: str | None = None,
        author: str | None = None,
        since: str | None = None,
        until: str | None = None,
        per_page: int = 30,
        page: int = 1,
        *,
        save_as: str,
    ) -> pd.DataFrame | None:
        """List commits in a GitHub repository.

        Args:
            owner (str): Repository owner
            repo (str): Repository name
            sha (Optional[str]): SHA or branch to start listing commits from
            path (Optional[str]): Only commits containing this file path
            author (Optional[str]): GitHub login or email address
            since (Optional[str]): ISO 8601 date string
            until (Optional[str]): ISO 8601 date string
            per_page (int): Results per page (max 100)
            page (int): Page number
            save_as (str): Variable name to store the commit list

        Returns:
            pd.DataFrame: Commits as a DataFrame

        """
        code = f'{save_as} = list_commits_impl("{owner}", "{repo}"'
        if sha:
            code += f', "{sha}"'
        else:
            code += ", None"
        if path:
            code += f', "{path}"'
        else:
            code += ", None"
        if author:
            code += f', "{author}"'
        else:
            code += ", None"
        if since:
            code += f', "{since}"'
        else:
            code += ", None"
        if until:
            code += f', "{until}"'
        else:
            code += ", None"
        code += f", {per_page}, {page})\n{save_as}"

        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    def _get_commit_details_impl(
        owner: str, repo: str, commit_sha: str
    ) -> pd.DataFrame:
        """Get detailed information about a specific commit.

        Args:
            owner (str): Repository owner
            repo (str): Repository name
            commit_sha (str): Commit SHA

        Returns:
            pd.DataFrame: Detailed commit information

        """
        session = _get_github_session()

        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"

        response = session.get(url)
        response.raise_for_status()

        commit = response.json()

        commit_data = commit["commit"]
        stats = commit.get("stats", {})

        row = {
            "sha": commit["sha"],
            "message": commit_data["message"],
            "author_name": commit_data["author"]["name"],
            "author_email": commit_data["author"]["email"],
            "author_date": commit_data["author"]["date"],
            "committer_name": commit_data["committer"]["name"],
            "committer_email": commit_data["committer"]["email"],
            "committer_date": commit_data["committer"]["date"],
            "additions": stats.get("additions", 0),
            "deletions": stats.get("deletions", 0),
            "total_changes": stats.get("total", 0),
            "files_changed": len(commit.get("files", [])),
            "html_url": commit["html_url"],
            "owner": owner,
            "repo": repo,
        }

        return pd.DataFrame([row])

    def _analyze_file_commits_around_issue_impl(
        owner: str,
        repo: str,
        issue_number: int,
        file_paths: list | None = None,
        days_before: int = 7,
        days_after: int = 1,
    ) -> pd.DataFrame:
        """Analyze commits to specific files around the time an issue was created.

        Args:
            owner (str): Repository owner
            repo (str): Repository name
            issue_number (int): Issue number to analyze around
            file_paths (Optional[list]): List of file paths to analyze (if None, analyzes all commits)
            days_before (int): Number of days before issue creation to look
            days_after (int): Number of days after issue creation to look

        Returns:
            pd.DataFrame: Commits around the issue timeframe

        """

        session = _get_github_session()

        issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        issue_response = session.get(issue_url)
        issue_response.raise_for_status()
        issue = issue_response.json()

        issue_created = dateutil.parser.parse(issue["created_at"])
        since_date = issue_created - timedelta(days=days_before)
        until_date = issue_created + timedelta(days=days_after)

        all_commits = []

        if file_paths:
            for file_path in file_paths:
                commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
                params = {
                    "path": file_path,
                    "since": since_date.isoformat(),
                    "until": until_date.isoformat(),
                    "per_page": 100,
                }

                response = session.get(commits_url, params=params)
                response.raise_for_status()

                commits = response.json()
                for commit in commits:
                    commit_data = commit["commit"]
                    all_commits.append(
                        {
                            "sha": commit["sha"],
                            "message": commit_data["message"],
                            "author_name": commit_data["author"]["name"],
                            "author_email": commit_data["author"]["email"],
                            "author_date": commit_data["author"]["date"],
                            "committer_date": commit_data["committer"]["date"],
                            "file_path": file_path,
                            "html_url": commit["html_url"],
                            "issue_number": issue_number,
                            "issue_created_at": issue["created_at"],
                            "days_from_issue": (
                                dateutil.parser.parse(commit_data["author"]["date"])
                                - issue_created
                            ).days,
                            "owner": owner,
                            "repo": repo,
                        }
                    )
        else:
            commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
            params = {
                "since": since_date.isoformat(),
                "until": until_date.isoformat(),
                "per_page": 100,
            }

            response = session.get(commits_url, params=params)
            response.raise_for_status()

            commits = response.json()
            for commit in commits:
                commit_data = commit["commit"]
                all_commits.append(
                    {
                        "sha": commit["sha"],
                        "message": commit_data["message"],
                        "author_name": commit_data["author"]["name"],
                        "author_email": commit_data["author"]["email"],
                        "author_date": commit_data["author"]["date"],
                        "committer_date": commit_data["committer"]["date"],
                        "file_path": None,
                        "html_url": commit["html_url"],
                        "issue_number": issue_number,
                        "issue_created_at": issue["created_at"],
                        "days_from_issue": (
                            dateutil.parser.parse(commit_data["author"]["date"])
                            - issue_created
                        ).days,
                        "owner": owner,
                        "repo": repo,
                    }
                )

        if not all_commits:
            return pd.DataFrame(
                columns=[
                    "sha",
                    "message",
                    "author_name",
                    "author_email",
                    "author_date",
                    "file_path",
                    "issue_number",
                    "days_from_issue",
                    "owner",
                    "repo",
                ]
            )

        return pd.DataFrame(all_commits)

    _SHELL.push(
        {
            "get_commit_details_impl": _get_commit_details_impl,
            "analyze_file_commits_around_issue_impl": _analyze_file_commits_around_issue_impl,
        }
    )

    @app.tool()
    async def get_commit_details(
        owner: str, repo: str, commit_sha: str, *, save_as: str
    ) -> pd.DataFrame | None:
        """Get detailed information about a specific commit.

        Args:
            owner (str): Repository owner
            repo (str): Repository name
            commit_sha (str): Commit SHA
            save_as (str): Variable name to store the commit details

        Returns:
            pd.DataFrame: Detailed commit information

        """
        code = f'{save_as} = get_commit_details_impl("{owner}", "{repo}", "{commit_sha}")\n{save_as}'
        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

    @app.tool()
    async def analyze_file_commits_around_issue(
        owner: str,
        repo: str,
        issue_number: int,
        file_paths: str | None = None,
        days_before: int = 7,
        days_after: int = 1,
        *,
        save_as: str,
    ) -> pd.DataFrame | None:
        """Analyze commits to specific files around the time an issue was created.

        Args:
            owner (str): Repository owner
            repo (str): Repository name
            issue_number (int): Issue number to analyze around
            file_paths (Optional[str]): Comma-separated list of file paths to analyze (if None, analyzes all commits)
            days_before (int): Number of days before issue creation to look
            days_after (int): Number of days after issue creation to look
            save_as (str): Variable name to store the analysis results

        Returns:
            pd.DataFrame: Commits around the issue timeframe

        """
        file_list = None
        if file_paths:
            file_list = [path.strip() for path in file_paths.split(",")]

        code = f'{save_as} = analyze_file_commits_around_issue_impl("{owner}", "{repo}", {issue_number}'
        if file_list:
            code += f", {file_list}"
        else:
            code += ", None"
        code += f", {days_before}, {days_after})\n{save_as}"

        execution_result = await run_code_in_shell(code)
        return execution_result.result if execution_result else None

else:
    logger.info("GitHub PAT token not detected - GitHub tools will not be registered")
