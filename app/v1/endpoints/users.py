from app.db.utils import *
from app.models.user import users
from app.schemas.user import UserCreate
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter()


@router.post("/register")
async def register_user(user: UserCreate):
    query = users.select().where(users.c.username == user.username)

    if len(await database.fetch_all(query)) > 0:
        raise HTTPException(status_code=409, detail="Username already in use.")

    query = users.insert().values(username=user.username, password=get_password_hash(user.password))
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