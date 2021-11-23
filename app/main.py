from fastapi import FastAPI, Depends

from app.core.db.database import *
from app.v1.endpoints import auth, users

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(users.router, prefix="/api/v1", tags=["v1"])

app.include_router(auth.router, prefix="/api/v1", tags=["v1"])


@app.get("/")
async def root():
    return {418: {"description": "Yes, quite scrumptious."}}