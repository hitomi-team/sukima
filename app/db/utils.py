from datetime import datetime, timedelta
from typing import AsyncIterator, Optional

import app.models.user as models
import app.schemas.user as schemas
from app.core.config import settings
from app.db.database import async_session
from app.schemas.token import TokenData
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=settings.TOKEN_URL)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthRequest(BaseModel):
    key: str


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


def get_password_hash(plain):
    return pwd_context.hash(plain)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with async_session() as session:
        try:
            yield session
        except Exception as e:
            raise e
        finally:
            await session.close()


async def get_user_by_username(session: AsyncSession, username: str) -> models.User:
    return (await session.execute(select(models.User).where(models.User.username == username))).scalars().first()


async def get_user_by_email(session: AsyncSession, email: str) -> models.User:
    return (await session.execute(select(models.User).where(models.User.email == email))).scalars().first()


async def create_user(session: AsyncSession, user: schemas.UserCreate) -> models.User:
    hashed_password = get_password_hash(user.password)
    db_user = models.User(username=user.username, password=hashed_password, email=user.email)

    session.add(db_user)
    await session.commit()
    await session.refresh(db_user)

    return db_user


async def authenticate_user(session: AsyncSession, username: str, password: str):
    user = await get_user_by_username(session, username)

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
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    return encoded_jwt


async def get_current_user(session: AsyncSession = Depends(get_session), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")

        if username is None:
            raise credentials_exception

        token_data = TokenData(username=username)

    except JWTError:
        raise credentials_exception

    user = await get_user_by_username(session, username=token_data.username)

    if user is None:
        raise credentials_exception

    return user


async def get_current_approved_user(current_user: schemas.User = Depends(get_current_user)):
    if not current_user.permission_level > 0:
        raise HTTPException(status_code=400, detail="Not approved.")

    return current_user
