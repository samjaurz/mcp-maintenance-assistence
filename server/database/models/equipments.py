from datetime import datetime
from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from . import Base


class Equipment(Base):
    __tablename__ = "equipments"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    location: Mapped[str] = mapped_column(String)
    area: Mapped[str] = mapped_column(String)
    type_category: Mapped[str] = mapped_column(String)
    status: Mapped[bool] = mapped_column(String)
    image: Mapped[str] = mapped_column(String)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    items: Mapped[list["EquipmentItem"]] = relationship("EquipmentItem", back_populates="equipment",
                                                            cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return (
            f"<Equipment "
            f"id={self.id}, "
            f"name={self.name}, "
            f"location={self.location}, "
            f"area={self.area}, "
            f"type_category={self.type_category}, "
            f"status={self.status}, "
            f"image={self.image}, "
            f"added_at={self.added_at}, "
            f"updated_at={self.updated_at}>"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "location": self.location,
            "area": self.area,
            "type_category": self.type_category,
            "status": self.status,
            "image": self.image,
            "added_at": self.added_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
