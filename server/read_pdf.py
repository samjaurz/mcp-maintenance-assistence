import os
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import datetime
from llama_cpp import Llama


model_embed = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)
chunks_list = []

MANUALS_FOLDER = "../manuals/"


llama_model_path = "../LLM/llama-7b.ggmlv3.q4_0.bin"
llm = Llama(model_path=llama_model_path)


def reading_file(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text



def process_chunks(text, source="manual.pdf", chunk_size=400, overlap=50):
    global chunks_list, index
    chunks = []
    start = 0
    chunk_id = len(chunks_list)

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(chunk_text)

        chunk_data = {
            "id": chunk_id,
            "text": chunk_text,
            "source": source,
            "type": type,
            "date_added": datetime.datetime.now().isoformat()
        }
        chunks_list.append(chunk_data)


        embedding = model_embed.encode([chunk_text]).astype('float32')
        index.add(embedding)

        start += chunk_size - overlap
        chunk_id += 1

    return chunks


def process_manuals(folder=MANUALS_FOLDER):
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)
            text = reading_file(path)
            process_chunks(text, source=filename)
            print(f"Procesado {filename}")



def search(query, top_k=3):
    query_emb = model_embed.encode([query]).astype('float32')
    D, I = index.search(query_emb, top_k)
    results = [chunks_list[i] for i in I[0]]
    return results



def ask_llama(query, top_chunks):
    context = "\n".join([c['text'] for c in top_chunks])
    prompt = f"Usa la información siguiente para responder la pregunta de manera clara y concisa:\n\n{context}\n\nPregunta: {query}\nRespuesta:"

    response = llm(prompt, max_tokens=512)

    return response['choices'][0]['text']


if __name__ == "__main__":
    process_manuals()

    print(f"\nNúmero total de chunks: {len(chunks_list)}")
    print(f"Número total de embeddings en FAISS: {index.ntotal}")


    query = "Dime qué hay sobre la batería"

    top_chunks = search(query, top_k=2)

    answer = ask_llama(query, top_chunks)

    print("RESPUESTA:",answer)
