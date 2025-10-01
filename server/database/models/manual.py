from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from . import Base


class Manual(Base):
    __tablename__ = "manuals"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    source_url: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunk: Mapped[list["Chunk"]] = relationship(
        back_populates="manual", order_by="Chunk.id", cascade="all, delete"
    )

    def __repr__(self) -> str:
        return (
            f"<Manual "
            f"id={self.id}, "
            f"name={self.name}, "
            f"source_url={self.source_url}, "
            f"category={self.category}, "
            f"added_at={self.added_at}>"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "source_url": self.source_url,
            "category": self.category,
            "added_at": self.added_at.isoformat(),
        }
