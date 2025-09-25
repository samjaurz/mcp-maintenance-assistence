from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi import Depends
from sqlalchemy.orm import Session
from server.modules.asking_claud import AskingCloud
from server.repositories.manual_repository import ManualRepository
from server.database.db_session import get_db
from server.modules.processing_pdf import ProcessorPDF
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

@app.get("/manuals")
def get_all_manuals(db_session: Session = Depends(get_db)):
    return ManualRepository(db_session).get_all_manuals()

@app.post("/manuals/upload")
async def get_all_manuals(file: UploadFile = File(...)):
    with open(f"manuals/{file.filename}", "wb") as buffer:
        buffer.write(await file.read())

    processor = ProcessorPDF(pdf_path=f"manuals/{file.filename}")
    processor.divide_pdf_in_chunks()

    return {"filename": file.filename}