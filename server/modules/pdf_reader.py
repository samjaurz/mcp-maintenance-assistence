from typing import Dict, Any
import pdfplumber
import fitz
from fastapi import UploadFile
import io
CID_PATTERN = "(cid:"


def extract_page_data(file: UploadFile) -> Dict[str, Any]:
    file_bytes = file.file.read()
    try:
        pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        print(f"Error al abrir el PDF con PyMuPDF: {e}")
        return {}

    pages_info_list = []

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            num_pages = min(len(pdf.pages), len(pdf_doc))

            for i in range(num_pages):
                page_number = i + 1

                page = pdf.pages[i]
                page_text = page.extract_text() or ""
                has_error = CID_PATTERN in page_text

                fitz_page = pdf_doc[i]
                image_list = fitz_page.get_images(full=False)
                has_images = len(image_list) > 0

                page_info = {
                    "page_number": page_number,
                    "extracted_text": page_text,
                    "has_error": has_error,
                    "images": has_images,
                }
                pages_info_list.append(page_info)
    except Exception as e:
        print(f"Error al abrir el PDF con pdfplumber: {e}")
        return {}

    pdf_doc.close()

    result = {
        "pages_total": num_pages,
        "source_filename": file.filename,
        "pages_info": pages_info_list,
    }

    return result
