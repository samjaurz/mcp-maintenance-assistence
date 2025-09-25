import faiss
import numpy as np
import os

class FaissModule:
    def __init__(self,
                 dimension: int = 384,
                 bin_path: str = "/Users/sam/Desktop/github/mpc_maintenance_assistence/server/llm/faiss_index.bin"
                 ):
        self.dimension = dimension
        self.bin_path = bin_path
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
    def selecting_path(self):
        if os.path.exists(self.bin_path):
            self.index = faiss.read_index(self.bin_path)
        self.save_bin()

    def load_bin(self):
        self.index = faiss.read_index(self.bin_path)

    def save_bin(self):
        return faiss.write_index(self.index, self.bin_path)

    def add_vector(self, embedding: np.ndarray, chunk_id: int,):
        return self.index.add_with_ids(embedding, np.array([chunk_id]))

    def search(self, query_emb: np.ndarray, top_k: int):
        D, I = self.index.search(query_emb, top_k)
        return [int(idx) for idx in I[0] if idx != -1]
