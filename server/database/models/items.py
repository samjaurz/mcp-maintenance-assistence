from datetime import datetime
from sqlalchemy import DateTime, String, func, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from . import Base


class Item(Base):
    __tablename__ = "items"
    id: Mapped[int] = mapped_column(primary_key=True)
    item_code: Mapped[str] = mapped_column(String)
    model: Mapped[str] = mapped_column(String)
    brand: Mapped[str] = mapped_column(String)
    manufacturer: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String)
    price : Mapped[float] = mapped_column(Float)
    quantity: Mapped[int] = mapped_column(Integer)
    bin_location: Mapped[str] = mapped_column(String)
    status: Mapped[bool] = mapped_column(String)
    image: Mapped[str] = mapped_column(String)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    equipments: Mapped[list["EquipmentItem"]] = relationship("EquipmentItem", back_populates="item",
                                                        cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return (
            f"<Item "
            f"id={self.id}, "
            f"item_code={self.item_code}, "
            f"model={self.model}, "
            f"brand={self.brand}, "
            f"manufacturer={self.manufacturer}, "
            f"category={self.category}, "
            f"price={self.price}, "
            f"quantity={self.quantity}, "
            f"bin_location={self.bin_location}, "
            f"status={self.status}, "
            f"image={self.image}, "
            f"added_at={self.added_at}, "
            f"updated_at={self.updated_at}>"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "item_code": self.item_code,
            "model": self.model,
            "brand": self.brand,
            "manufacturer": self.manufacturer,
            "category": self.category,
            "price": self.price,
            "quantity": self.quantity,
            "bin_location": self.bin_location,
            "status": self.status,
            "image": self.image,
            "added_at": self.added_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
