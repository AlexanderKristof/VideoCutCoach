from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import app_paths

# Helyi SQLite adatbázis (később kicserélhető PostgreSQL-re). Fejlesztői módban a régi
# "sqlite:///./tactical_app.db", alkalmazás-módban az adatmappában (lásd app_paths.py).
SQLALCHEMY_DATABASE_URL = app_paths.database_url()
if app_paths.app_mode():
    os.makedirs(os.path.dirname(app_paths.db_path()), exist_ok=True)

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