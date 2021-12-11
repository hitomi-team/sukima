from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.v1.routes.models as models
import app.v1.routes.users as users
from app.core.config import settings
from app.db.database import database


def get_application():
    _app = FastAPI(title=settings.PROJECT_NAME)

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return _app


app = get_application()

app.include_router(users.router, prefix="/api/v1", tags=["v1"])
app.include_router(models.router, prefix="/api/v1", tags=["v1"])


@app.on_event("startup")
async def startup():
    # async with engine.connect() as conn:
    #    await conn.run_sync(metadata.create_all)

    await database.connect()


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
