import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from server.repositories.chunk_repository import ChunkRepository
from server.repositories.manual_repository import ManualRepository
from server.database.db_session import SessionLocal
import numpy as np

class ProcessorPDF:

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def reading_pdf(self) -> str:
        db_session = SessionLocal()
        extracted_text = ""
        pdf_name = self.pdf_path.split("/")[-1]
        manual_exists = ManualRepository(db_session).get_manual_by_name(pdf_name)
        if manual_exists:
            return "manual_exists"
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_text += page_text + "/f"
        ManualRepository(db_session).add_manual(pdf_name, self.pdf_path)
        return extracted_text

    def divide_pdf_in_chunks(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        text = self.reading_pdf()
        chunks = splitter.split_text(text)
        self.embedding_chunks(chunks)
        return chunks

    def embedding_chunks(self, chunks: list):
        db_session = SessionLocal()
        model = SentenceTransformer("all-MiniLM-L6-v2")
        dimension = 384
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        pdf_name = self.pdf_path.split("/")[-1]
        manual_exists = ManualRepository(db_session).get_manual_by_name(pdf_name)
        for chunk in chunks:
            create_chunk = ChunkRepository(db_session).add_chunk(
                text=chunk,
                source=self.pdf_path,
                manual_id= manual_exists[0].id,
                )
            embedding = model.encode([chunk]).astype("float32")
            index.add_with_ids(embedding, np.array([create_chunk.id]))
            faiss.write_index(index, "/Users/sam/Desktop/github/mpc_maintenance_assistence/server/llm/faiss_index.bin")

        return {"index": index, "chunks": chunks}