from server.modules.asking_claud import AskingCloud

def test_anthropic_response():
    cloud = AskingCloud()
    query = "¿Cuáles son las especificaciones de entrada y salida de la válvula?"
    top_chunks = cloud.search(query, top_k=2)
    answer = cloud.ask_anthropic(query, top_chunks)

    print("\nRESPUESTA:", answer)



