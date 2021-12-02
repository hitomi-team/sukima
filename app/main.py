from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.core.db.database import *
from app.v1.endpoints import auth, users

app = FastAPI()

app.include_router(users.router, prefix="/api/v1", tags=["v1"])

app.include_router(auth.router, prefix="/api/v1", tags=["v1"])

# TODO: Make the origins configurable
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {418: {"description": "Yes, quite scrumptious."}}
