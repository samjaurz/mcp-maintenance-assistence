import os

import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from server.database.db_session import SessionLocal
from server.modules.embedding_module import EmbeddingModule
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
        # Combinar contexto
        context = "\n".join([c.text for c in top_chunks if c.text.strip()])

        # Middleware: decidir estructura del prompt según el tipo de pregunta
        if context:
            # Si parece que la pregunta es sobre un error, darle formato especial
            if "error" in query.lower():
                prompt = f"""
    Responde de manera clara y precisa siguiendo estas reglas:

    1. Si el contexto contiene pasos para resolver el error, listalos en orden.
    2. Indica claramente cuál es el error mencionado.
    3. Usa SOLO la información del contexto.
    4. No agregues explicaciones adicionales fuera de los pasos.

    Contexto:
    {context}

    Pregunta:
    {query}

    Respuesta:
    """
            else:
                # Otro tipo de pregunta
                prompt = f"""
    Responde usando SOLO la información del contexto. Sé claro y conciso.

    Contexto:
    {context}

    Pregunta:
    {query}

    Respuesta:
    """
        else:
            # No hay información relevante
            prompt = f"""
    No se encontró información en los manuales.  
    Pero basado en LLM: [respuesta del modelo]

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
