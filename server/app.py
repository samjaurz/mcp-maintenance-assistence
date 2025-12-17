from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database.db_session import get_session
from modules.asking_claud import AskingCloud
from modules.pdf_processor import ProcessorPDF
from repositories.manual_repository import ManualRepository
from fastapi.responses import FileResponse
from dotenv import load_dotenv
load_dotenv()
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

class PageInfoRequest(BaseModel):
    pdf_filename: str
    page_number: int


@app.post("/send-text")
def send_text(data: TextRequest):
    cloud = AskingCloud()
    top_chunks = cloud.search(data.text, top_k=3)
    answer = cloud.ask_anthropic(data.text, top_chunks)
    return {"message": answer}


@app.get("/manuals")
def get_all_manuals(db_session: Session = Depends(get_session)):
    return ManualRepository(db_session).get_all_manuals()


@app.post("/manuals/upload")
async def upload_pdf(
        file: UploadFile = File(...), db_session: Session = Depends(get_session)
):
    result = ProcessorPDF(file).process_and_embedding_pdf(db_session)
    return result


@app.delete("/manuals/{manual_id}")
async def delete_pdf(manual_id: int, db_session: Session = Depends(get_session)):
    return ManualRepository(db_session).delete_manual(manual_id)
