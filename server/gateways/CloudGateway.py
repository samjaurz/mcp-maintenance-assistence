from sqlalchemy.orm import Session
from server.repositories.chunk_repository import ChunkRepository
from server.llm_motors.abstract_llm import InterfaceLLM
from server.modules.embedding_module import EmbeddingModule
from server.modules.faiss_module import FaissModule
from typing import List
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LLMGateway:

    def __init__(self, llm_motor: InterfaceLLM):
        self.llm_motor = llm_motor
        self.faiss = FaissModule()
        self.embedding = EmbeddingModule()

    def search(self, session: Session, query: str, top_k: int = 3) -> List[int]:
        query_embedding = self.embedding.vectorize_text(query)
        valid_ids = self.faiss.search(query_embedding, top_k)

        if not valid_ids:
            return []

        top_chunks = ChunkRepository(session).get_chunks_ids(valid_ids)
        return top_chunks

    def ask_with_rag(self, session: Session, prompt: str) -> str:
        top_chunks = self.search(session, prompt)
        if not top_chunks:
            return "No se encontró información relevante para su consulta."
        response_text = self.llm_motor.ask_model(prompt, top_chunks)
        return response_text
