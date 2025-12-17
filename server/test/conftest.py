import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app import app
from database.models import Base
from database.db_session import get_session


TEST_DATABASE_URL = "sqlite:///:memory"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_session] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture
def real_pdf():
    path = "/Users/sam/Desktop/github/mpc_maintenance_assistence/manuals/variador_frecuencia_fr-f800.pdf"

    with open(path, "rb") as f:
        f.seek(0)
        yield f


