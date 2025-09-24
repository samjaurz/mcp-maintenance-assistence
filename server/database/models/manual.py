from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, DateTime, func
import datetime
from . import Base


class Manual(Base):
    __tablename__ = "manuals"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    source: Mapped[str] = mapped_column(String)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    chunk: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="manual", order_by="Chunk.id"
    )

    def __repr__(self) -> str:
        return (
            f"<Manual "
            f"id={self.id}, "
            f"name={self.name}, "
            f"source={self.source}, "
            f"added_at={self.added_at}>"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "added_at": self.added_at.isoformat(),
        }
