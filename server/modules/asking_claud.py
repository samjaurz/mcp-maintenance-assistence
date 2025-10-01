import os

import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from server.database.db_session import SessionLocal
from server.modules.embedding_module import EmbeddingModule
from server.modules.faiss_module import FaissModule
from server.repositories.chunk_repository import ChunkRepository

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
api_key_anthropic = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key_anthropic)


class AskingCloud:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.embedding = EmbeddingModule()
        self.faiss = FaissModule()

    def search(self, query, top_k=3):
        query_embedding = self.embedding.vectorize_text(query)
        query_vector_list = query_embedding.flatten().tolist()
        with SessionLocal() as db:
            relevant_chunks = ChunkRepository(db).search_similar_chunks(
            query_embedding=query_vector_list,
            top_k=top_k
        )

        return relevant_chunks

    def ask_anthropic(self, query, top_chunks):
        # if not top_chunks:
        #     return "No encontré información relevante en la base de datos."

        context = "\n".join([c.text for c in top_chunks])
        prompt = f"""
        Responde de manera clara y concisa siguiendo estas reglas estrictas:

        1. Si el contexto contiene información relevante para responder la pregunta, usa SOLO esa información.
        2. Si el contexto está vacío o no contiene información relevante, responde EXACTAMENTE en el siguiente formato:

        No se encontró información en los manuales.  
        Pero basado en LLM: [respuesta del modelo]

        No agregues introducciones ni frases adicionales fuera de este formato.

        Contexto:
        {context}

        Pregunta:
        {query}

        Respuesta:
        """

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text
