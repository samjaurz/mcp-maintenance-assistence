from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, DateTime, func
import datetime
from . import Base


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(primary_key=True)
    text: Mapped[str] = mapped_column(String)
    source: Mapped[str] = mapped_column(String)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return (
            f"<Chunk "
            f"id={self.id}, "
            f"text={self.text}, "
            f"source={self.source}, "
            f"added_at={self.added_at}>"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "added_at": self.added_at.isoformat(),
        }
