from datetime import datetime

from sqlalchemy.orm import Session

from database.models.chunk import Chunk
from typing import List

class ChunkRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_chunk(self, text: str, embedding: list[float], manual_id: int) -> Chunk:
        now = datetime.now()
        chunk = Chunk(text=text, embedding=embedding, manual_id=manual_id, added_at=now)
        self.session.add(chunk)
        self.session.commit()
        return chunk

    def get_chunks_ids(self, ids: list[int]):
        return self.session.query(Chunk).filter(Chunk.id.in_(ids)).all()

    def search_similar_chunks(self, query_embedding: List[float], top_k: int = 5) -> List[type[Chunk]]:
        results = (
            self.session.query(Chunk)
            .order_by(Chunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
            .all()
        )

        return results

    def delete_chunk(self, chunk_id: int) -> bool:
        chunk = self.session.get(Chunk, chunk_id)
        if chunk:
            self.session.delete(chunk)
            self.session.commit()
            return True
        return False
