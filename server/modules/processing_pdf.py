import pdfplumber
import os
import json
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
        history = self.read_history()
        all_files = {}
        for filename in os.listdir(self.path_folders):
            if filename.endswith(".pdf") and filename:
                path = os.path.join(self.path_folders, filename)
                text = self.reading_pdf(path)
                all_files[filename] = text
                self.save_history(filename)

        return all_files

    def read_history(self):
        history_path = "/Users/sam/Desktop/github/mpc_maintenance_assistence/manuals/history_pdf.txt"
        with open(history_path, 'r', encoding='utf-8') as file:
            history = json.loads(file)
            return history

    def save_history(self, filename: str) -> None:
        history_path = "/Users/sam/Desktop/github/mpc_maintenance_assistence/manuals/history_pdf.txt"
        history =  self.read_history()
        if filename not in history["manuals"]:
            history["manuals"].append(filename)
        with open(history_path, 'ra', encoding='utf-8') as file:
            history = json.load(file)





