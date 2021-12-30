from app.api.v1.endpoints import models, soft_prompts, users
from fastapi import APIRouter

api_router = APIRouter()

api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(soft_prompts.router, prefix="/softprompts", tags=["softprompts"])
