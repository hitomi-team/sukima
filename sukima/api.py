from functools import wraps
from fastapi import APIRouter, Request
from util import Util
from pydantic import BaseModel

# NOTE: This is incomplete until the webapp uses a factory pattern.
class API:
    def __init__(self, version='v1', app=None, config=None ):
        self.version = version
        self.app = app
        self.config = config

    class AuthRequest(Request):
        pass

    router = APIRouter()

    @router.get("/v1/create_key", tags=["model"])
    async def create_key(self, request: Request, ):
        pass