import json
import os
import tempfile
from typing import List
import io
import pdfplumber
from fastapi import HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session
from modules.embedding_module import EmbeddingModule
from repositories.chunk_repository import ChunkRepository
from repositories.manual_repository import ManualRepository

class ProcessorPDF:
    def __init__(self, file: UploadFile):
        self.file = file
        self.embedding = EmbeddingModule()

    def process_and_embedding_pdf(self, session: Session) -> dict:
        pdf_name = self.file.filename
        manual = ManualRepository(session).get_manual_by_name(pdf_name)
        if manual:
            raise HTTPException(
                status_code=409, detail=f"The manual '{pdf_name}' already exists."
            )
        try:
            text = self.extract_text()
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

        chunks = self._divide_pdf_in_chunks(text)
        self._embedding_chunks(session, chunks)

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "PDF processed successfully"}),
        }

    def extract_text(self) -> str:
        extracted_text = ""

        try:
            self.file.file.seek(0)
            with pdfplumber.open(self.file.file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"

        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {e}")

        return extracted_text

    def _divide_pdf_in_chunks(
        self, text: str, chunk_size: int = 400, chunk_overlap: int = 50
    ) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)

    def _embedding_chunks(self, session: Session, chunks: List[str]):

        pdf_name = self.file.filename
        chunk_repository = ChunkRepository(session)
        manual_repository = ManualRepository(session)

        new_manual = manual_repository.add_manual(
            name=pdf_name,
            category="timer",
            source_url=self.file.filename
        )

        successful_chunks = []

        try:
            for chunk in chunks:
                embedding_array = self.embedding.vectorize_text(chunk)
                embedding_list = embedding_array.flatten().tolist()

                new_chunk = chunk_repository.add_chunk(
                    text=chunk,
                    embedding=embedding_list,
                    manual_id=new_manual.id,
                )
                successful_chunks.append(new_chunk)

            session.commit()

        except Exception as e:
            session.rollback()
            raise e

        return {
            "manual_id": new_manual.id,
            "chunk_count": len(successful_chunks),
            "status": "success"
        }
