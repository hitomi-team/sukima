from app.db.utils import *
from app.schemas.user import UserCreate
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.logger import logger
import logging

router = APIRouter()
logger.setLevel(logging.INFO)


@router.post("/register")
async def register_user(user: UserCreate, session: AsyncSession = Depends(get_session)):
    user = await get_user_by_email(session, user.email)
    logger.info('test log')

    if not user:
        query = users.insert().values(username=user.username, password=get_password_hash(user.password))
        logger.info('inserting new user')
        await database.execute(query)

    return {"Successfully created user."}


@router.post("/token")
async def generate_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    expiration = timedelta(days=settings.ACCESS_TOKEN_EXPIRATION)
    token = create_access_token({"sub": user["username"]}, expiration)

    return {"token": token, "token_type": "bearer"}
