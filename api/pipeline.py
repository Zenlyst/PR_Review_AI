"""
Review pipeline — orchestrates the full review flow for one PR.

Called as a background task from the webhook endpoint.
"""

import logging

from sqlalchemy import select

from api.database import async_session
from api.diff_parser import extract_before_after
from api.github_service import GitHubService
from api.model_service import model_service
from api.models import Review

logger = logging.getLogger(__name__)

# Code length bounds (same as training filters)
MIN_CODE_LINES = 5
MAX_CODE_LINES = 200

NO_ISSUE_PHRASES = ["no issues", "looks good", "lgtm", "no problems"]


async def review_pipeline(
    installation_id: int,
    repo_full_name: str,
    pr_number: int,
    base_sha: str,
    head_sha: str,
) -> None:
    """
    Full pipeline:
        1. Authenticate as GitHub App installation
        2. Fetch changed Python files
        3. For each file: get before/after → generate review
        4. Post review comments to PR
        5. Save reviews to database
    """
    github = GitHubService()
    try:
        token = await github.get_installation_token(installation_id)
        files = await github.get_pr_files(token, repo_full_name, pr_number)

        logger.info(
            "Reviewing %s#%d — %d Python file(s)", repo_full_name, pr_number, len(files)
        )

        review_comments: list[dict] = []

        for file_info in files:
            # Skip deleted files
            if file_info["status"] == "removed":
                continue

            file_path = file_info["filename"]

            # Fetch before/after code
            before_code, after_code = await extract_before_after(
                github, token, repo_full_name, base_sha, head_sha, file_info
            )

            # Skip files outside training code-length bounds
            line_count = len(after_code.splitlines())
            if line_count < MIN_CODE_LINES or line_count > MAX_CODE_LINES:
                logger.debug("Skipping %s (%d lines, outside bounds)", file_path, line_count)
                continue

            # Generate review
            review_text = model_service.review(before_code, after_code, file_path)

            # Skip "no issues" responses
            if _is_no_issues(review_text):
                logger.debug("No issues found for %s, skipping", file_path)
                continue

            review_comments.append({
                "path": file_path,
                "body": review_text,
                "line": 1,
            })

            # Persist to database
            await _save_review(
                repo_full_name=repo_full_name,
                pr_number=pr_number,
                file_path=file_path,
                before_code=before_code,
                after_code=after_code,
                review_comment=review_text,
                commit_sha=head_sha,
            )

        # Post all comments as one review
        if review_comments:
            await github.post_review(
                token, repo_full_name, pr_number, head_sha, review_comments
            )
        else:
            logger.info("No actionable reviews for %s#%d", repo_full_name, pr_number)

    except Exception:
        logger.exception("Review pipeline failed for %s#%d", repo_full_name, pr_number)
    finally:
        await github.close()


def _is_no_issues(review_text: str) -> bool:
    """Check if the model output indicates no issues found."""
    lower = review_text.lower()
    return any(phrase in lower for phrase in NO_ISSUE_PHRASES)


async def _save_review(
    repo_full_name: str,
    pr_number: int,
    file_path: str,
    before_code: str,
    after_code: str,
    review_comment: str,
    commit_sha: str,
) -> None:
    """Save a review record to the database."""
    async with async_session() as session:
        review = Review(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            file_path=file_path,
            before_code=before_code,
            after_code=after_code,
            review_comment=review_comment,
            commit_sha=commit_sha,
        )
        session.add(review)
        await session.commit()


async def get_reviews(
    repo_full_name: str, pr_number: int | None = None
) -> list[dict]:
    """Query review history from the database."""
    async with async_session() as session:
        stmt = select(Review).where(Review.repo_full_name == repo_full_name)
        if pr_number is not None:
            stmt = stmt.where(Review.pr_number == pr_number)
        stmt = stmt.order_by(Review.created_at.desc())

        result = await session.execute(stmt)
        reviews = result.scalars().all()
        return [
            {
                "id": r.id,
                "pr_number": r.pr_number,
                "file_path": r.file_path,
                "review_comment": r.review_comment,
                "commit_sha": r.commit_sha,
                "created_at": r.created_at.isoformat(),
            }
            for r in reviews
        ]
