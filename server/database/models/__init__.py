from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


from .chunk import Chunk  # noqa: E402
from .manual import Manual  # noqa: E402

__all__ = [
    "Base",
    "Chunk",
    "Manual",
]
