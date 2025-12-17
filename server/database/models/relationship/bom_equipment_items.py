from sqlalchemy import Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .. import Base


class EquipmentItem(Base):
    __tablename__ = 'equipment_item'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    equipment_id: Mapped[int] = mapped_column(ForeignKey('equipments.id'), nullable=False)
    item_id: Mapped[int] = mapped_column(ForeignKey('items.id'), nullable=False)

    # Relationships
    equipment: Mapped["Equipment"] = relationship("Equipment", back_populates="items")
    item: Mapped["Item"] = relationship("Item", back_populates="equipments")

    def __repr__(self):
        return (f"<EquipmentItem(id={self.id}, "
                f"equipment_id={self.equipment_id}, "
                f"item_id={self.item_id})>")
