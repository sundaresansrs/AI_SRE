"""MCP server exposing read-only GitHub repository inspection tools."""

import os
from datetime import datetime
from itertools import islice
from typing import Any

from github import Auth, Github
from github.GithubException import GithubException
from mcp.server.fastmcp import FastMCP

TOKEN = os.environ["GITHUB_PAT"]
mcp = FastMCP("github")
github = Github(auth=Auth.Token(TOKEN))


def _timestamp(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _github_error(error: GithubException) -> str:
    status = getattr(error, "status", None)
    message = getattr(error, "data", None) or str(error)
    if isinstance(message, dict):
        message = message.get("message", str(error))
    return f"GitHub API error ({status}): {message}" if status else f"GitHub API error: {message}"


def _repo(repo_full_name: str) -> Any:
    return github.get_repo(repo_full_name)


@mcp.tool()
def list_workflow_runs(repo_full_name: str, limit: int = 10) -> list[dict[str, Any]] | str:
    """List recent GitHub Actions workflow runs for a repository."""
    try:
        if limit < 1:
            return "Error: limit must be at least 1."
        runs = _repo(repo_full_name).get_workflow_runs()
        return [
            {
                "id": run.id,
                "status": run.status,
                "conclusion": run.conclusion,
                "created_at": _timestamp(run.created_at),
            }
            for run in islice(runs, limit)
        ]
    except GithubException as error:
        return _github_error(error)
    except Exception as error:
        return f"GitHub error: {error}"


@mcp.tool()
def list_open_issues(repo_full_name: str) -> list[dict[str, Any]] | str:
    """List open issues, excluding pull requests, in a repository."""
    try:
        issues = _repo(repo_full_name).get_issues(state="open")
        return [
            {
                "number": issue.number,
                "title": issue.title,
                "created_at": _timestamp(issue.created_at),
            }
            for issue in issues
            if issue.pull_request is None
        ]
    except GithubException as error:
        return _github_error(error)
    except Exception as error:
        return f"GitHub error: {error}"


@mcp.tool()
def get_issue(repo_full_name: str, issue_number: int) -> dict[str, Any]:
    """Get issue details, or a clear error dictionary if it does not exist."""
    try:
        issue = _repo(repo_full_name).get_issue(number=issue_number)
        return {
            "title": issue.title,
            "body": issue.body,
            "state": issue.state,
            "labels": [label.name for label in issue.labels],
        }
    except GithubException as error:
        return {"error": _github_error(error)}
    except Exception as error:
        return {"error": f"GitHub error: {error}"}


@mcp.tool()
def list_open_pull_requests(repo_full_name: str) -> list[dict[str, Any]] | str:
    """List open pull requests in a repository."""
    try:
        pulls = _repo(repo_full_name).get_pulls(state="open")
        return [
            {
                "number": pull.number,
                "title": pull.title,
                "created_at": _timestamp(pull.created_at),
            }
            for pull in pulls
        ]
    except GithubException as error:
        return _github_error(error)
    except Exception as error:
        return f"GitHub error: {error}"


if __name__ == "__main__":
    mcp.run()