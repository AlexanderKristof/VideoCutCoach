from datetime import datetime, timedelta
from jose import jwt, JWTError
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from database import get_db
from models import User

import app_paths

# A JWT aláírókulcs: TACTICAL_SECRET_KEY környezeti változó, különben az első induláskor generált,
# az adatmappában tárolt véletlen kulcs (secret.key; fejlesztői módban ./.tacanaly_secret.key).
SECRET_KEY = app_paths.load_or_create_secret_key()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 hétig érvényes token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def _password_bytes(password: str) -> bytes:
    # A bcrypt legfeljebb 72 bájtot használ (a korábbi passlib-es hash-ek is így készültek);
    # a bcrypt 5 a hosszabbra hibát dobna, ezért itt vágjuk le.
    return (password or "").encode("utf-8")[:72]

def hash_password(password: str) -> str:
    # Közvetlenül a bcrypt csomaggal (a passlib 1.7.4 a bcrypt 5-tel nem működik: minden
    # regisztráció hibával elszállt). A régi passlib-es "$2b$…" hash-ek ugyanígy ellenőrizhetők.
    return bcrypt.hashpw(_password_bytes(password), bcrypt.gensalt()).decode("ascii")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(_password_bytes(plain_password), (hashed_password or "").encode("ascii"))
    except ValueError:
        return False

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# EZ A FÜGGVÉNY VÉDI AZ ENDPOINT-OKAT:
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Aktiválási hiba vagy érvénytelen munkamenet.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user