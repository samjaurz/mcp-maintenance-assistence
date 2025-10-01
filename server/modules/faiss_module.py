import os

import faiss
import numpy as np

DIMENSION = 768
BIN_PATH = (
    "/Users/sam/Desktop/github/mpc_maintenance_assistence/server/llm/faiss_index.bin"
)


class FaissModule:
    def __init__(self):
        self.dimension = DIMENSION
        self.bin_path = BIN_PATH
        self.index = None
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.bin_path):
            self.index = faiss.read_index(self.bin_path)
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))

    def save_bin(self):
        return faiss.write_index(self.index, self.bin_path)

    def add_vector(
        self,
        embedding: np.ndarray,
        chunk_id: int,
    ):
        return self.index.add_with_ids(embedding, np.array([chunk_id]))

    def search(self, query_emb: np.ndarray, top_k: int):
        D, I = self.index.search(query_emb, top_k)
        return [int(idx) for idx in I[0] if idx != -1]
