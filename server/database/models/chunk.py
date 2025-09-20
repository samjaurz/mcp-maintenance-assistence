from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, Boolean, ForeignKey

from . import Base


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    status: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    user: Mapped["User"] = relationship(back_populates="task")

    def __repr__(self) -> str:
        return f"<Task id={self.id} name={self.name}>"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "user_id": self.user_id,
        }