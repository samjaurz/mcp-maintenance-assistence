from sentence_transformers import SentenceTransformer
import faiss


class FaisGateway:

    def __init__(self, bin_path: str, sentence_transformer: SentenceTransformer):
        self.bin_path = bin_path
        self.main_index = faiss.read_index(bin_path)
        self.model = sentence_transformer
        print("📂 Índice FAISS cargado con", self.main_index.ntotal, "embeddings")

    def index_search(self, prompt, top_k=3) -> list:
        query_emb = self.model.encode([prompt]).astype("float32")
        D, I = self.main_index.search(query_emb, top_k)

        return [int(idx) for idx in I[0] if idx != -1]
