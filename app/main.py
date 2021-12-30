from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.v1.api import api_router

app = FastAPI(
    title=settings.PROJECT_NAME
)

if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

app.include_router(api_router, prefix="/api/v1", tags=["v1"])

# TODO: add access control to this, because the rest of the app allows for "private" soft prompts
app.mount("/storage", StaticFiles(directory=settings.STORAGE_PATH), name="storage")

@app.get("/")
async def root():
    return 'Sometimes I dream about cheese.'