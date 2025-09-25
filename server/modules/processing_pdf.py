import pdfplumber
import json

from fastapi import UploadFile, File
from langchain.text_splitter import RecursiveCharacterTextSplitter
from server.repositories.chunk_repository import ChunkRepository
from server.repositories.manual_repository import ManualRepository
from server.database.db_session import SessionLocal
from server.modules.faiss_module import FaissModule
from server.modules.embedding_module import EmbeddingModule


class ProcessorPDF:
    def __init__(self):
        self.pdf_path = "/Users/sam/Desktop/github/mpc_maintenance_assistence/manuals/macbook-info.pdf"
        self.faiss = FaissModule()
        self.embedding = EmbeddingModule()
        self.session = SessionLocal()

    def reading_pdf(self) -> dict:
        extracted_text = ""
        pdf_name = self.pdf_path.split("/")[-1]
        manual_exists = ManualRepository(self.session).get_manual_by_name(pdf_name)
        if manual_exists:
            return {
                "statusCode": 404,
                "body": json.dumps({
                    "message": "Manual already in database",
                })}

        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "/f"
        ManualRepository(self.session).add_manual(pdf_name, self.pdf_path)
        return extracted_text

    def divide_pdf_in_chunks(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        text = self.reading_pdf()
        chunks = splitter.split_text(text)
        self.embedding_chunks(chunks)
        return chunks

    def add_chunk_database(self, chunks):


    def add_vector(self, chunk: int):
        embedding = self.embedding.vectorize_many_texts(chunk)
        self.faiss.add_vector(embedding, create_chunk.id)
        return
    def embedding_chunks(self, chunks: list):
        db_session = SessionLocal()
        pdf_name = self.pdf_path.split("/")[-1]
        manual_exists = ManualRepository(db_session).get_manual_by_name(pdf_name)
        for chunk in chunks:
            create_chunk = ChunkRepository(db_session).add_chunk(
                text=chunk,
                source=self.pdf_path,
                manual_id= manual_exists[0].id,
                )
            embedding = self.embedding.vectorize_many_texts(chunk)
            self.faiss.add_vector(embedding, create_chunk.id)

        self.faiss.save_bin()

        return { "index": self.faiss.index, "chunks": chunks}