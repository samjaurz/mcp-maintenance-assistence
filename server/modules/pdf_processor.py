import json
import os
import tempfile
from typing import List

import pdfplumber
from fastapi import HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from server.modules.embedding_module import EmbeddingModule
from server.modules.faiss_module import FaissModule
from server.repositories.chunk_repository import ChunkRepository
from server.repositories.manual_repository import ManualRepository


class ProcessorPDF:
    def __init__(self, file: UploadFile):
        self.file = file
        self.faiss = FaissModule()
        self.embedding = EmbeddingModule()

    def process_and_embedding_pdf(self, session: Session) -> dict:
        pdf_name = self.file.filename
        manual = ManualRepository(session).get_manual_by_name(pdf_name)
        if manual:
            raise HTTPException(
                status_code=409, detail=f"The manual '{pdf_name}' already exists."
            )
        try:
            text = self._extract_text_from_pdf()
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

        chunks = self._divide_pdf_in_chunks(text)
        self._embedding_chunks(session, chunks)

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "PDF processed successfully"}),
        }

    def _extract_text_from_pdf(self) -> str:
        extracted_text = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(self.file.file.read())
            temp_path = temp_file.name

        try:
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "/f"
        except Exception:
            raise ValueError("Error extracting text from PDF.")
        finally:
            os.remove(temp_path)

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

        # TODO: The logic for creating the manual doesn’t convince me, but I need the manual ID for the 1:n relationship with the chunks.
        #  Ideally, the manual should be created at the end, once everything has been generated. maybe add pages,chunk_totals, chunk_start, chunk_finish

        new_manual = ManualRepository(session).add_manual(pdf_name, self.file.filename)

        for chunk in chunks:
            create_chunk = chunk_repository.add_chunk(
                text=chunk,
                source=pdf_name,
                manual_id=new_manual.id,
            )
            embedding = self.embedding.vectorize_text(chunk)
            self.faiss.add_vector(embedding, create_chunk.id)
        self.faiss.save_bin()
