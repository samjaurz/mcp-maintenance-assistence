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
        valid_ids = self.faiss.search(query_embedding, top_k)
        if not valid_ids:
            print("⚠️ No se encontraron resultados en FAISS")
            return []

        with SessionLocal() as db:
            chunks = ChunkRepository(db).get_chunks_ids(valid_ids)

        if chunks:
            print("Chunks recuperados de la BD:", [c.id for c in chunks])
        else:
            print(
                "⚠️ La consulta a la BD no recuperó ningún chunk. Valid IDs:", valid_ids
            )

        return chunks

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
