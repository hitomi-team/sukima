import ray
from ray import serve
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1.api import api_router

app = FastAPI(
    title=settings.PROJECT_NAME
)
ray.init(address="auto", namespace="raytest")
serve.start(detached=True)

if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS]
    )

app.include_router(api_router, prefix="/api/v1", tags=["v1"])


@app.get("/")
async def root():
    return 'Sometimes I dream about cheese.'


# Ray cluster launcher wen
@serve.deployment(route_prefix="/raytest", num_replicas=2)
@serve.ingress(app)
class FAPIWrapper:
    @app.get("/")
    def root(self):
        return "I hate cheese!"
