import os
import pdfplumber
import fitz  # PyMuPDF
import json
from typing import List, Dict, Any

CID_PATTERN = "(cid:"


def extract_page_data(pdf_path: str) -> Dict[str, Any]:
    """
    Extrae texto de cada página de un PDF con la estructura solicitada.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: No se encontró el archivo: {pdf_path}")
        return {}

    try:
        pdf_doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error al abrir el PDF con PyMuPDF (fitz): {e}")
        return {}

    pages_info_list = []

    with pdfplumber.open(pdf_path) as pdf:
        num_pages = min(len(pdf.pages), len(pdf_doc))

        for i in range(num_pages):
            page_number = i + 1

            # Extracción de texto y detección de CID
            page = pdf.pages[i]
            page_text = page.extract_text() or ""
            has_error = CID_PATTERN in page_text

            # Detección de Imágenes
            fitz_page = pdf_doc[i]
            image_list = fitz_page.get_images(full=False)
            has_images = len(image_list) > 0

            page_info = {
                "page_number": page_number,
                "extracted_text": page_text,
                "has_error": has_error,
                "images": has_images
            }
            pages_info_list.append(page_info)

    pdf_doc.close()

    # Estructura final como la solicitas
    result = {
        "pages_total": num_pages,
        "source_url": pdf_path,
        "pages_info": pages_info_list  # Lista de todas las páginas
    }

    return result


if __name__ == "__main__":
    TEST_PDF_PATH = "/Users/sam/Desktop/github/mpc_maintenance_assistence/manuals/timer_aluminio_wc5a30102.pdf"
    OUTPUT_JSON_NAME = "pdf_extraction_results_clean.json"

    print(f"--- Iniciando extracción de texto y clasificación para: {TEST_PDF_PATH} ---")

    resulta = extract_page_data(TEST_PDF_PATH)

    if resulta:
        print("\n--- Clasificación de Extracción ---")
        for page_info in resulta["pages_info"]:
            status = "FALLO (CID)" if page_info["has_error"] else "OK"
            img_status = "SÍ" if page_info["images"] else "NO"
            print(f"Página {page_info['page_number']:02}: {status} | Imágenes: {img_status}")

        # Guardar JSON
        with open(OUTPUT_JSON_NAME, "w", encoding="utf-8") as f:
            json.dump(resulta, f, ensure_ascii=False, indent=4)

        print(f"\n--- Proceso Finalizado ---")
        print(f"Resultados guardados en {OUTPUT_JSON_NAME}")
        print(f"Total páginas procesadas: {resulta['pages_total']}")
    else:
        print("No se pudieron extraer datos del PDF.")