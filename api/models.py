"""
SQLAlchemy ORM models for review history storage.
"""

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from api.database import Base


class Review(Base):
    """Stores each AI-generated code review."""

    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_full_name: Mapped[str] = mapped_column(String(255), index=True)
    pr_number: Mapped[int] = mapped_column(Integer, index=True)
    file_path: Mapped[str] = mapped_column(Text)
    before_code: Mapped[str] = mapped_column(Text)
    after_code: Mapped[str] = mapped_column(Text)
    review_comment: Mapped[str] = mapped_column(Text)
    commit_sha: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
