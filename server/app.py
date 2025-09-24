from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.database.models import Chunk
from server.gateways.ClaudeGateway import ClaudeGateway
from server.modules.asking_claud import AskingCloud
from server.modules.assistant import Assistant
from server.repositories.chunk_repository import ChunkRepository

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home():
    return {"message": "¡Hola, FastAPI !"}

class TextRequest(BaseModel):
    text: str
@app.post("/send-text")
def send_text(data: TextRequest):
    cloud = AskingCloud()
    top_chunks = cloud.search(data.text, top_k=2)
    print(f"Buscando: {data.text}")
    answer = cloud.ask_anthropic(data.text, top_chunks)
    return {"message": answer}

@app.post("/send-text2")
def send_text(data: TextRequest):
    assistant = Assistant(llm_gateway=ClaudeGateway(
        client=Antropik(),
        chunk_repo=ChunkRepository(session="")
    ))
    answer = assistant.ask_question(data.text)
    return {"message": answer}
