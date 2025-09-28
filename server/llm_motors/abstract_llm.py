from abc import ABC, abstractmethod
from typing import List


class InterfaceLLM(ABC):

    @abstractmethod
    def ask_model(self, prompt: str, top_chunks: List[int]) -> str:
        pass
