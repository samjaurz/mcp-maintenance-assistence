from pathlib import Path
from typing import Any, Dict, List, Union

from llama_cpp import Llama

from .abstract_llm import InterfaceLLM

LLAMA_MODEL_PATH = Path("../LLM/llama-7b.ggmlv3.q4_0.bin")
MAX_TOKENS = 512


class MotorLLAMA(InterfaceLLM):
    def __init__(self, path_model: Union[str, Path]):
        self.motor = Llama(model_path=str(path_model))
        self.max_tokens = MAX_TOKENS

    def _build_prompt(self, prompt: str, context: str) -> str:
        return (
            f"Use the following information to answer the question clearly and concisely:"
            f"\n\nContext:\n{context}\n\n"
            f"Question:\n{prompt}"
            f"\nAnswer:"
        )

    def ask_model(self, prompt: str, top_chunks: List[Dict[str, Any]]) -> str:
        context = "\n".join([c["text"] for c in top_chunks])
        prompt_final = self._build_prompt(prompt, context)

        response = self.motor(prompt_final, max_tokens=self.max_tokens)

        return response["choices"][0]["text"]
