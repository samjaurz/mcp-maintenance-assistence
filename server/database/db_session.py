from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from server.database.models import Base as MainBase

DATABASE_URL = "postgresql://root:root@localhost:5432/mcp-db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = MainBase

def get_session():
    with Session(engine) as session:
        yield session
