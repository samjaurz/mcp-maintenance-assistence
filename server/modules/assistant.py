from server.gateways import LLMGateway


class Assistant:

    def __init__(self, llm_gateway: LLMGateway):
        self.llm_gateway = llm_gateway

    def ask_question(self, prompt):
        answer = self.llm_gateway.search(prompt)
        return answer
