from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Helyi SQLite adatbázis fejlesztéshez (később kicserélhető PostgreSQL-re)
SQLALCHEMY_DATABASE_URL = "sqlite:///./tactical_app.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Adatbázis munkamenet-kezelő függőség a FastAPI-hoz
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()