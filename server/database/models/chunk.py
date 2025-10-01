from datetime import datetime
from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from . import Base


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(String)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    manual_id: Mapped[int] = mapped_column(ForeignKey("manuals.id", ondelete="CASCADE"))

    manual: Mapped["Manual"] = relationship(back_populates="chunk")

    def __repr__(self) -> str:
        return (
            f"<Chunk "
            f"id={self.id}, "
            f"text={self.text}, "
            f"added_at={self.added_at}>"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "added_at": self.added_at.isoformat(),
        }
