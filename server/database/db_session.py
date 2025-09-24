from functools import wraps
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from server.database.models import Base as MainBase

DATABASE_URL = "postgresql://root:root@localhost:5432/mcp-db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = MainBase

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def with_db_session(func):
    """
    A decorator to handle opening and closing the database session.
    """
    @wraps(func)
    def wrapper(event, context):
        session = None
        try:
            # Pass the session as a keyword argument to the function
            if "session" in context:
                session = context["session"]
            else:
                session = next(get_db())
            context['session'] = session
            # Call the decorated function
            return func(event, context)
        except Exception as e:
            raise e
        finally:
            if session:
                session.close()
    return wrapper