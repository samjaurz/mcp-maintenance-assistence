from datetime import datetime
from typing import Type

from sqlalchemy.orm import Session

from database.models.manual import Manual


class ManualRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_manual(self, name: str, source_url: str, category: str) -> Manual:
        now = datetime.now()
        manual = Manual(name=name, source_url=source_url, category= category, added_at=now)
        self.session.add(manual)
        self.session.commit()
        return manual

    def get_manual_by_name(self, name: str) -> list[Type[Manual]] | None:
        return self.session.query(Manual).filter(Manual.name.ilike(f"%{name}%")).first()

    def get_all_manuals(self) -> list[Type[Manual]] | None:
        return self.session.query(Manual).order_by(Manual.id).all()

    def delete_manual(self, manual_id: int) -> bool:
        manual = self.session.query(Manual).filter_by(id=manual_id).first()
        if manual:
            self.session.delete(manual)
            self.session.commit()
            return True
        return False
