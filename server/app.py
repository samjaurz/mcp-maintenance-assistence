from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from server.modules.pdf_reader import extract_page_data
from server.database.db_session import get_session
# from server.modules.asking_claud import AskingCloud
from server.modules.pdf_processor import ProcessorPDF
from server.modules.renderizado_pagina import get_rendered_page_url
from server.repositories.manual_repository import ManualRepository
from fastapi.responses import FileResponse
import time
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


# Modelo para la nueva ruta de renderizado de página
class PageInfoRequest(BaseModel):
    pdf_filename: str  # Ejemplo: "mitsubishi-a800-manual.pdf"
    page_number: int  # Ejemplo: 30


@app.post("/send-text")
def send_text(data: TextRequest):
    pass
    # cloud = AskingCloud()
    # top_chunks = cloud.search(data.text, top_k=3)
    # print(f"Buscando: {data.text}")
    # # answer = cloud.ask_anthropic(data.text, top_chunks)
    # return {"message": top_chunks}


@app.get("/manuals")
def get_all_manuals(db_session: Session = Depends(get_session)):
    return ManualRepository(db_session).get_all_manuals()


@app.post("/manuals/upload")
async def upload_pdf(
        file: UploadFile = File(...), db_session: Session = Depends(get_session)
):
    start = time.time()
    result = extract_page_data(file)
    print("⏱️ Tiempo de procesamiento:", time.time() - start, "segundos")
    return result


@app.delete("/manuals/{manual_id}")
async def delete_pdf(manual_id: int, db_session: Session = Depends(get_session)):
    return ManualRepository(db_session).delete_manual(manual_id)

@app.get("/manuals/get_manual")
async def get_manual():
    file_path = "/Users/sam/Desktop/github/mpc_maintenance_assistence/server/modules/pdf_extraction_results_clean.json"
    return FileResponse(file_path, media_type="application/json", filename="manual.json")


from fastapi import Request
import json


@app.post("/manuals/retrieve_page")
async def retrieve_page(request: Request):
    """
    Recibe el nombre del PDF y el número de página en el cuerpo JSON,
    renderiza la página a una imagen y devuelve la URL pública.
    """
    try:
        # Obtener el cuerpo JSON crudo
        body = await request.json()
        print("📥 Body recibido:", body)

        # Extraer los campos manualmente
        pdf_filename = body.get("pdf_filename")
        page_number = body.get("page_number")

        # Validar que existan los campos requeridos
        if not pdf_filename or not page_number:
            raise HTTPException(
                status_code=422,
                detail="Faltan campos requeridos: pdf_filename y page_number"
            )

        print(f"🔍 Procesando: {pdf_filename}, página {page_number}")

        # La función get_rendered_page_url ahora se importa de server.modules.renderizado_pagina
        url = get_rendered_page_url(pdf_filename, page_number)

        if url:
            print(f"✅ URL generada: {url}")
            return {"url": url}

        # Si la URL es None, algo falló
        raise HTTPException(
            status_code=400,
            detail="Error al renderizar la página. Verifique el nombre del archivo y el número de página."
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="JSON inválido")
    except Exception as e:
        print(f"❌ Error general: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@app.post("/manuals/process_page/")
async def process_page(request: Request):
    data = await request.json()  # recibes el body tal cual
    print("📥 Payload recibido:", data)  # lo imprimes en consola

    return {
        "message": "Página recibida en backend",
        "data": data
    }


@app.post("/manuals/process_selection/")
async def process_selection(request: Request):
    try:
        data = await request.json()   # 👈 aquí obtienes el body tal cual
        print("📥 Datos recibidos:", data)

        # ejemplo de acceso
        page_number = data.get("page_number")
        pdf_filename = data.get("pdf_filename")
        selection = data.get("selection", {})
        image_url = data.get("image_url")

        print(f"Página: {page_number}, Archivo: {pdf_filename}")
        print(f"Selección: {selection}")
        print(f"Imagen: {image_url}")

        return {
            "status": "ok",
            "message": "Selección recibida correctamente",
            "data": data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))