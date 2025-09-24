import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import datetime
import anthropic
import os
from dotenv import load_dotenv
from server.database.db_session import with_db_session

model_embed = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384
index = faiss.IndexFlatL2(dimension)

chunks_list = []


load_dotenv()
api_key_anthropic = os.getenv("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(
    api_key=api_key_anthropic,
)

def reading_file(pdf_path):
    text = ""
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
            "date_added": datetime.datetime.now().isoformat(),
        }
        chunks_list.append(chunk_data)


        embedding = model_embed.encode([chunk_text]).astype("float32")
        index.add(embedding)

        start += chunk_size - overlap
        chunk_id += 1

    return chunks


def search(query, top_k=3):
    query_emb = model_embed.encode([query]).astype("float32")
    D, I = index.search(query_emb, top_k)
    results = [chunks_list[i] for i in I[0]]
    return results


def ask_anthropic(query, top_chunks):
    context = "\n".join([c["text"] for c in top_chunks])
    prompt = f"""Usa la información siguiente para responder de manera clara y concisa:

    Contexto:
    {context}

    Pregunta:
    {query}

    Respuesta:"""

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


if __name__ == "__main__":

    pdf_path = "../manuals/valvulas.pdf"
    text = reading_file(pdf_path)
    process_chunks(text, source=pdf_path)

    print(f"\nNúmero total de chunks: {len(chunks_list)}")
    print(f"Número total de embeddings en FAISS: {index.ntotal}")

    query = "Cuales son las especificaciones de entrada y salida de la valvula"
    top_chunks = search(query, top_k=2)
    answer = ask_anthropic(query, top_chunks)

    print("RESPUESTA:", answer)
