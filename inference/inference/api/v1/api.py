from inference.api.v1.endpoints import models
from fastapi import APIRouter

api_router = APIRouter()

api_router.include_router(models.router, prefix="/models", tags=["models"])
