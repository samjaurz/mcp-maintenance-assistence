from datetime import datetime

from sqlalchemy.orm import Session

from server.database.models.chunk import Chunk


class ChunkRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_chunk(self, text: str, source: str, manual_id: int) -> Chunk:
        now = datetime.now()
        chunk = Chunk(text=text, source=source, manual_id=manual_id, added_at=now)
        self.session.add(chunk)
        self.session.commit()
        return chunk

    def get_chunks_ids(self, ids: list[int]):
        return self.session.query(Chunk).filter(Chunk.id.in_(ids)).all()

    def delete_chunk(self, chunk_id: int) -> bool:
        chunk = self.session.get(Chunk, chunk_id)
        if chunk:
            self.session.delete(chunk)
            self.session.commit()
            return True
        return False
