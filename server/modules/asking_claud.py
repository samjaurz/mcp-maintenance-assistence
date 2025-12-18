import os

import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from database.db_session import SessionLocal
from modules.embedding_module import EmbeddingModule
from repositories.chunk_repository import ChunkRepository

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

    def search(self, query, top_k=2):
        query_embedding = self.embedding.vectorize_text(query)
        query_vector_list = query_embedding.flatten().tolist()
        with SessionLocal() as db:
            relevant_chunks = ChunkRepository(db).search_similar_chunks(
                query_embedding=query_vector_list,
                top_k=top_k
            )

        return relevant_chunks

    def ask_anthropic(self, query, top_chunks):

        context = "\n".join([c.text for c in top_chunks if c.text.strip()])

        prompt = f"""
        Eres un ingeniero mecánico senior especializado en diagnóstico de fallas.

        INFORMACIÓN DE MANUALES:
        {context}

        FALLA REPORTADA:
        {query}

        Responde como ingeniero priorizando lo siguiente en orden no 
        hace falta que pongas los numeros solo que se entienda la separacion si no se encuentra 
        nada en el contexto relevante indicar en refrencia a los manuales que no hay nada, y si en base a que manual:
        
        1. Diagnóstico probable (con probabilidad %)
        2. Pasos para verificar
        3. Solución recomendada
        4. Tiempo y costo estimado
        5. Referencias a manuales
        
        Respuesta: 
        """

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text
