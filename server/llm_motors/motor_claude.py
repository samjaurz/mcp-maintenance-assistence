from .abstract_llm import InterfaceLLM
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()
api_key_anthropic = os.getenv("ANTHROPIC_API_KEY")

MODEL_NAME = "claude-3-haiku-20240307"
MAX_TOKENS = 512


class MotorClaude(InterfaceLLM):
    def __init__(self):
        self.model_name = "claude-3-haiku-20240307"
        self.max_tokens = 512
        self.client = anthropic.Anthropic(api_key=api_key_anthropic)

    def _build_prompt(self, prompt, context):
        return f"""Usa la información siguiente para responder de manera clara y concisa:

                Contexto:
                {context}

                Pregunta:
                {prompt}

                Respuesta:"""

    def ask_model(self, prompt, top_chunks):
        context = "\n".join([c.text for c in top_chunks])
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": self._build_prompt(prompt, context)}],
        )
        return response.content[0].text
