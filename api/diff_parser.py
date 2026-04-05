"""
Extract before/after code from PR file changes.

Strategy: fetch full file content at base_sha (before) and head_sha (after).
This matches the training data format better than parsing unified diffs.
"""

from api.github_service import GitHubService


async def extract_before_after(
    github: GitHubService,
    token: str,
    repo_full_name: str,
    base_sha: str,
    head_sha: str,
    file_info: dict,
) -> tuple[str, str]:
    """
    Fetch file content at base and head commits.

    Args:
        file_info: A single entry from GitHub's PR files endpoint.

    Returns:
        (before_code, after_code)

    Edge cases:
        - New file (status="added"): before_code is empty
        - Renamed file: uses previous_filename for before
    """
    file_path = file_info["filename"]
    status = file_info["status"]

    # New file — no before version
    if status == "added":
        before_code = ""
    else:
        before_path = file_info.get("previous_filename", file_path)
        before_code = await github.get_file_content(
            token, repo_full_name, base_sha, before_path
        )

    after_code = await github.get_file_content(
        token, repo_full_name, head_sha, file_path
    )

    return before_code, after_code
