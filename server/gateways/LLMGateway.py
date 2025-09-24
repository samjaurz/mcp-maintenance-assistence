class LLMGateway:

    def search(self, prompt: str):
        raise NotImplementedError

    def ask_model(self):
        raise NotImplementedError