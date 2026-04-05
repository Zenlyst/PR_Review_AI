"""
GitHub webhook signature verification and event parsing.
"""

import hashlib
import hmac

from fastapi import HTTPException, Request

from api.config import settings


def verify_signature(payload_body: bytes, signature_header: str | None) -> None:
    """
    Verify the HMAC-SHA256 signature from GitHub.
    Raises HTTPException(401) if invalid.
    """
    if not signature_header:
        raise HTTPException(status_code=401, detail="Missing signature header")

    expected = hmac.new(
        settings.github_webhook_secret.encode(),
        payload_body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(f"sha256={expected}", signature_header):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")


def parse_pr_event(request: Request, payload: dict) -> dict | None:
    """
    Parse a webhook payload and return PR metadata if it's a reviewable event.

    Returns None for events we don't care about (non-PR, closed, etc.).
    Returns a dict with keys needed by the review pipeline.
    """
    event_type = request.headers.get("X-GitHub-Event")
    if event_type != "pull_request":
        return None

    action = payload.get("action")
    if action not in ("opened", "synchronize"):
        return None

    pr = payload["pull_request"]
    return {
        "installation_id": payload["installation"]["id"],
        "repo_full_name": payload["repository"]["full_name"],
        "pr_number": pr["number"],
        "base_sha": pr["base"]["sha"],
        "head_sha": pr["head"]["sha"],
    }
