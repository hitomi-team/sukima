from datetime import datetime, timedelta
from typing import Optional

from app.core.config import settings
from app.db.database import database
from app.db.schemas import users
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from passlib.context import CryptContext

# Avoid hardcoding this. Use an envvar or config or something. Or move this somewhere else.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/users/token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


def get_password_hash(plain):
    return pwd_context.hash(plain)


async def authenticate_user(username: str, password: str):
    query = users.select().where(users.c.username == username)
    user = await database.fetch_one(query)

    if not user:
        return False

    if not verify_password(password, user["password"]):
        return False

    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7)  # 7 day token expiration by default

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    return encoded_jwt
