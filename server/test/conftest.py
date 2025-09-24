import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from server.database.models import Base


@pytest.fixture(scope="function")
def db_session():
    engine = create_engine(os.environ.get("DATABASE_URL"))
    connection = engine.connect()
    transaction = connection.begin()

    SessionLocal = scoped_session(
        sessionmaker(autocommit=False, autoflush=False, bind=connection)
    )
    session = SessionLocal()

    Base.metadata.create_all(bind=engine)

    try:
        yield session
    finally:
        transaction.rollback()
        session.close()
        SessionLocal.remove()
        connection.close()
