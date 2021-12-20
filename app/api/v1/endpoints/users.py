from datetime import timedelta

import app.crud.user as crud
from app.api.deps import get_session
from app.core.config import settings
from app.core.security import create_access_token
from app.schemas.user import UserCreate
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.post("/register")
async def register_user(user: UserCreate, session: AsyncSession = Depends(get_session)):
    db_user = await crud.user.get_user_by_email(session, user.email)

    if not db_user:
        await crud.user.create_user(session, user)

    return {"Successfully created user."}


@router.post("/token")
async def generate_token(form_data: OAuth2PasswordRequestForm = Depends(), session: AsyncSession = Depends(get_session)):
    user = await crud.user.authenticate_user(session, form_data.username, form_data.password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    expiration = timedelta(days=settings.ACCESS_TOKEN_EXPIRATION)
    token = create_access_token({"sub": user.username}, expiration)

    return {"access_token": token, "token_type": "bearer"}
