import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import datetime
import os
import json

MANUALS_FOLDER = "../manuals/"

class PDFProcessor:
    def __init__(self, path_folders: str):
        self.path_folders = path_folders

    def reading_pdf(self, pdf_path: str) -> str:
        extracted_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "/f"
        return extracted_text

    def reading_files(self):
        all_files = {}
        for filename in os.listdir(self.path_folders):
            if filename.endswith(".pdf"):
                path = os.path.join(self.path_folders, filename)
                text = self.reading_pdf(path)
                all_files[filename] = text
        return all_files


