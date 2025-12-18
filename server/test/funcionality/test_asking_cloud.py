from modules.asking_claud import AskingCloud

def test_anthropic_response():
    cloud = AskingCloud()
    query = "dime 3 cosas de variadores"
    top_chunks = cloud.search(query, top_k=2)
    answer = cloud.ask_anthropic(query, top_chunks)

    print("\nRESPUESTA:", top_chunks)
    print(answer)

if __name__ == "__main__":
    test_anthropic_response()
