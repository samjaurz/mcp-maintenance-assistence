from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile

from server.database.db_session import get_session
from server.modules.pdf_processor import ProcessorPDF


def test_reading_files_real():
    pdf_path = (
        "/Users/sam/Desktop/github/mpc_maintenance_assistence/manuals/variador_frecuencia_fr-f800.pdf"
    )

    session: Session = next(get_session())

    try:
        with open(pdf_path, "rb") as f:
            upload_file = UploadFile(filename="variador_frecuencia_fr-f800.pdf", file=f)

            processor = ProcessorPDF(upload_file)

            result = processor.process_and_embedding_pdf(session)

            print(result)

    finally:
        # 5. Asegúrate de cerrar la sesión al finalizar la prueba, incluso si falla
        session.close()
