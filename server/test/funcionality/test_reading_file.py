from server.modules.processing_pdf import ProcessorPDF
from server.test.conftest import db_session
import json


def test_reading_files_real():
    pdf_path = (
        "/Users/sam/Desktop/github/mpc_maintenance_assistence/manuals/macbook-info.pdf"
    )

    text = ProcessorPDF(pdf_path)
    a = ProcessorPDF.divide_pdf_in_chunks(text)
    print(a)




