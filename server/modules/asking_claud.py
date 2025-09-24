from sentence_transformers import SentenceTransformer
import faiss
import anthropic
import os
from dotenv import load_dotenv
from server.database.db_session import SessionLocal
from server.repositories.chunk_repository import ChunkRepository

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()
api_key_anthropic = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key_anthropic)


class AskingCloud:
    def __init__(self,
                 faiss_index_path="/Users/sam/Desktop/github/mpc_maintenance_assistence/server/llm/faiss_index.bin"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        self.faiss_index_path = faiss_index_path

        if not os.path.exists(self.faiss_index_path):
            raise FileNotFoundError(f"No se encontró el índice FAISS en {self.faiss_index_path}")

        self.index = faiss.read_index(self.faiss_index_path)
        print("📂 Índice FAISS cargado con", self.index.ntotal, "embeddings")

    def search(self, query, top_k=3):
        query_emb = self.model.encode([query]).astype("float32")
        D, I = self.index.search(query_emb, top_k)
        valid_ids = [int(idx) for idx in I[0] if idx != -1]
        if not valid_ids:
            print("⚠️ No se encontraron resultados en FAISS")
            return []

        with SessionLocal() as db:
            chunks = ChunkRepository(db).get_chunks_ids(valid_ids)

        print(f"🔎 Consulta: '{query}'")
        print("IDs encontrados en FAISS:", valid_ids)

        if chunks:
            print("Chunks recuperados de la BD:", [c.id for c in chunks])
        else:
            print("⚠️ La consulta a la BD no recuperó ningún chunk. Valid IDs:", valid_ids)

        return chunks

    def ask_anthropic(self, query, top_chunks):
        if not top_chunks:
            return "No encontré información relevante en la base de datos."

        # Concatenar textos de los chunks
        context = "\n".join([c.text for c in top_chunks])
        prompt = f"""Usa la información siguiente para responder de manera clara y concisa:

        Contexto:
        {context}

        Pregunta:
        {query}

        Respuesta:"""

        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text
