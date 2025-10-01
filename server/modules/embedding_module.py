import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModule:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)

    def vectorize_text(self, text: str) -> np.ndarray:
        return self.model.encode([text]).astype("float32")

    def vectorize_many_texts(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts).astype("float32")
