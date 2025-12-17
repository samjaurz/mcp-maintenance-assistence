from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


from .chunk import Chunk  # noqa: E402
from .manual import Manual  # noqa: E402
from .items import Item  # noqa: E402
from .equipments import Equipment  # noqa: E402
from .relationship.bom_equipment_items import EquipmentItem  # noqa: E402

__all__ = [
    "Base",
    "Chunk",
    "Manual",
    "Item",
    "Equipment",
    "EquipmentItem",
]
