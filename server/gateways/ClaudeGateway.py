from server.gateways.FaisGateway import FaisGateway
from server.gateways.LLMGateway import LLMGateway
from server.repositories.chunk_repository import ChunkRepository
from  anthropic import Anthropic
from server.modules.faiss_module import FaissModule
class ClaudeGateway(LLMGateway):
    def __init__(self,
                 client: Anthropic,
                 chunk_repo: ChunkRepository,
                 fais_gw: FaisGateway
                 ):
        self.client = client
        self.chunk_repo = chunk_repo
        self.fais_gw = FaissModule()
        self.model_name = "claude-3-haiku-20240307"
        self.max_tokens = 512


    def search(self, prompt):
        valid_ids = self.fais_gw.index_search(prompt)
        chunks = self.chunk_repo.get_chunks_ids(valid_ids)
        print(f"🔎 Consulta: '{prompt}'")
        print("IDs encontrados en FAISS:", valid_ids)

        if chunks:
            print("Chunks recuperados de la BD:", [c.id for c in chunks])
        else:
            print("⚠️ La consulta a la BD no recuperó ningún chunk. Valid IDs:", valid_ids)

        return chunks


    def _build_prompt(self, prompt, context):
        return f"""Usa la información siguiente para responder de manera clara y concisa:

                Contexto:
                {context}

                Pregunta:
                {prompt}

                Respuesta:"""

    def ask_model(self, prompt, top_chunks):
        if not top_chunks:
            return "No encontré información relevante en la base de datos."

        context = "\n".join([c.text for c in top_chunks])
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content":  self._build_prompt(prompt, context)}],
        )
        return response.content[0].text