"""
GitHub App authentication and API client.

Handles:
- JWT creation from App private key
- Installation token exchange
- Fetching PR changed files and file content
- Posting PR review comments
"""

import logging
import time
from base64 import b64decode

import httpx
import jwt

from api.config import settings

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


class GitHubService:
    """Async GitHub API client authenticated as a GitHub App."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=GITHUB_API,
            headers={"Accept": "application/vnd.github+json"},
            timeout=30.0,
        )

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _create_jwt(self) -> str:
        """
        Create a short-lived JWT signed with the App's private key.
        Valid for 10 minutes (GitHub maximum).
        """
        now = int(time.time())
        payload = {
            "iat": now - 60,  # issued at (60s clock skew buffer)
            "exp": now + (10 * 60),  # expires in 10 minutes
            "iss": str(settings.github_app_id),
        }
        return jwt.encode(payload, settings.github_private_key, algorithm="RS256")

    async def get_installation_token(self, installation_id: int) -> str:
        """Exchange JWT for an installation access token (valid ~1 hour)."""
        jwt_token = self._create_jwt()
        resp = await self._client.post(
            f"/app/installations/{installation_id}/access_tokens",
            headers={"Authorization": f"Bearer {jwt_token}"},
        )
        resp.raise_for_status()
        token = resp.json()["token"]
        return token

    # ------------------------------------------------------------------
    # PR Data
    # ------------------------------------------------------------------

    async def get_pr_files(
        self, token: str, repo_full_name: str, pr_number: int
    ) -> list[dict]:
        """
        Fetch list of changed files in a PR.
        Returns only Python files (matching Phase 1 training scope).
        """
        resp = await self._client.get(
            f"/repos/{repo_full_name}/pulls/{pr_number}/files",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        files = resp.json()
        return [f for f in files if f["filename"].endswith(".py")]

    async def get_file_content(
        self, token: str, repo_full_name: str, ref: str, path: str
    ) -> str:
        """Fetch raw file content at a specific commit SHA."""
        resp = await self._client.get(
            f"/repos/{repo_full_name}/contents/{path}",
            params={"ref": ref},
            headers={"Authorization": f"Bearer {token}"},
        )
        resp.raise_for_status()
        data = resp.json()
        # GitHub returns base64-encoded content
        return b64decode(data["content"]).decode("utf-8")

    # ------------------------------------------------------------------
    # Post Review
    # ------------------------------------------------------------------

    async def post_review(
        self,
        token: str,
        repo_full_name: str,
        pr_number: int,
        commit_sha: str,
        comments: list[dict],
    ) -> None:
        """
        Post a PR review with inline file comments.

        Each comment: {"path": "file.py", "body": "review text", "line": 1}
        Event is always COMMENT — we never auto-approve.
        """
        body = {
            "commit_id": commit_sha,
            "event": "COMMENT",
            "body": "**PR Review AI** (automated code review)",
            "comments": comments,
        }
        resp = await self._client.post(
            f"/repos/{repo_full_name}/pulls/{pr_number}/reviews",
            headers={"Authorization": f"Bearer {token}"},
            json=body,
        )
        resp.raise_for_status()
        logger.info(
            "Posted review with %d comments on %s#%d",
            len(comments),
            repo_full_name,
            pr_number,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()
