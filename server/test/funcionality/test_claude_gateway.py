from server.gateways.ClaudeGateway import  ClaudeGateway

def test_anthropic_response():
    cloud = ClaudeGateway()
    query = "top 4 cosas mas relveantes"
    top_chunks = cloud.search(query, top_k=2)
    answer = cloud.ask_anthropic(query, top_chunks)
    print("\nRESPUESTA:", answer)


