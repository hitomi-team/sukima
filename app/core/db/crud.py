from datetime import timedelta, datetime
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from starlette import status

from app.core.config import SECRET_KEY
import app.core.db.models as models
import app.core.db.schemas as schemas
from app.core.db.database import SessionLocal
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session


ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

# Avoid hardcoding this. Use an envvar or config or something. Or move this somewhere else.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/token")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


def get_password_hash(plain):
    return pwd_context.hash(plain)


# Review FastAPI db docs
def get_user(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()


def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.User).offset(skip).limit(limit).all()


def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(username=user.username, password=hashed_password)

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return db_user


def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7)  # 7 day token expiration by default
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_approved_user(current_user: models.User = Depends(get_current_user)):
    if not current_user.approved:
        raise HTTPException(status_code=400, detail="Not approved.")
    return current_user


# Model names stored in DB are NOT model objects (even if we have ORM), they are names of models that a User can use
def get_models(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.ModelItem).offset(skip).limit(limit).all()


def create_model_item(item: schemas.ModelItemCreate, model_name: str):
    pass
