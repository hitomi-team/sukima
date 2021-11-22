import json

from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from typing import List, Optional

from auth import Auth
from gpthf import GPTHF
from util import Util

with open("config.json") as f:
    config = json.load(f)
version = 'v1'
if config["auth_enable"]:
    auth = Auth(config)
models = []


# Oh jeez that's a lot of models
class ModelGenArgs(BaseModel):
    max_length: int
    max_time: Optional[float] = None


class ModelSampleArgs(BaseModel):
    temp: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tfs: Optional[float] = None
    rep_p: Optional[float] = None
    rep_p_range: Optional[int] = None
    rep_p_slope: Optional[float] = None
    bad_words: List[str] = None
    bias_words: List[str] = None
    bias: Optional[float] = None


class ModelGenRequest(BaseModel):
    model: str
    prompt: str
    sample_args: ModelSampleArgs
    gen_args: ModelGenArgs


class ModelLoadRequest(BaseModel):
    model: str
    parallel: Optional[bool] = False
    sharded: Optional[bool] = False


class AuthRequest(BaseModel):
    key: str


app = FastAPI()


@app.get(f"/{version}/create_key", tags=["auth"], response_model=AuthRequest)
async def create_key():
    if not config["auth_enable"]:
        return Util.error(None, "Authentication disabled")
    return {"key": auth.create_key()}


@app.post(f"/{version}/delete_key", tags=["auth"])
async def delete_key(request: AuthRequest):
    if not config["auth_enable"]:
        return Util.error(None, "Authentication disabled")

    # No success or failure indicators.
    auth.delete_key(request.key)
    return Util.success("Key delete request processed")


# Get models
@app.get(f"/{version}/models", tags=["model"])
async def get_model_list():
    model_dict = {
        "models": {
        }
    }
    for model in models:
        model_dict["models"][model.model_name] = {'ready': True}
    return model_dict


# Load model
@app.post(f"/{version}/load", tags=["model"])
async def load_model(request: ModelLoadRequest):
    # Check that model exists
    if models is not None:
        for m in models:
            if m.model_name == request.model:
                return Util.error(None, "Model already loaded")
    try:
        model = GPTHF(model_name=request.model, parallelize=request.parallel, sharded=request.sharded)
        models.append(model)
        return Util.success("Loaded model")
    except:
        return Util.error(None, "Unsupported model")


# Delete model
@app.post(f"/{version}/delete", tags=["model"])
async def delete_model(request: ModelLoadRequest):
    for m in models:
        if m.model_name == request.model:
            models.remove(m)
            return Util.success("Deleted model")
    return Util.error(None, "Model not found")


# Generate from model
@app.post(f"/{version}/generate", tags=["model"])
async def generate(request: ModelGenRequest):
    for m in models:
        if m.model_name == request.model:
            try:
                return Util.completion(m.generate(request.dict()))
            except Exception as e:
                return Util.error(None, f'Invalid request body! \n{e}')

    return Util.error(None, "Model not found")
