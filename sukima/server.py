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
    max_length: float
    max_time: Optional[float] = None


class ModelSampleArgs(BaseModel):
    temp: float
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    tfs: Optional[float] = None
    rep_p: float
    rep_p_range: Optional[float] = None
    rep_p_slope: Optional[float] = None
    bad_words: List[str] = []
    bias_words: List[str] = []
    bias: float


class ModelReqArgs(BaseModel):
    prompt: str
    sample_args: ModelSampleArgs
    gen_args: ModelGenArgs


class ModelRequest(BaseModel):
    model: str
    parallel: Optional[bool] = False
    args: Optional[ModelReqArgs] = None


app = FastAPI()


@app.get(f"/{version}/create_key", tags=["auth"])
async def create_key():
    if not config["auth_enable"]:
        return Util.error(None, "Authentication disabled")
    return {
        "key": auth.create_key()
    }


@app.get(f"/{version}/delete_key", tags=["auth"])
async def delete_key(request: ModelRequest):
    if config["auth_enable"] == False:
        return Util.error(None, "Authentication disabled")

    auth.delete_key(request["key"])
    return Util.success("Key deleted")


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
@app.get(f"/{version}/load", tags=["model"])
async def load_model(request: ModelRequest):
    # Check that model exists
    if models is not None:
        for m in models:
            if m.model_name == request.model:
                return Util.error(None, "Model already loaded")
    try:
        model = GPTHF(model_name=request.model, parallelize=request.parallel)
        models.append(model)
        return Util.success("Loaded model")
    except:
        return Util.error(None, "Unsupported model")

# Delete model
@app.get(f"/{version}/delete", tags=["model"])
async def delete_model(request: ModelRequest):
    for m in models:
        if m.model_name == request.model:
            models.remove(m)
            return Util.success("Deleted model")
    return Util.error(None, "Model not found")

# Generate from model
@app.get(f"/{version}/generate", tags=["model"])
async def generate(request: ModelRequest):
    for m in models:
        if m.model_name == request.model:
            try:
                if request.args is None:
                    return Util.error(None, "No generation arguments specified")
                return Util.completion(m.generate(dict(request.args)))
            except Exception as e:
                return Util.error(None, 'Invalid request body!')

    return Util.error(None, "Model not found")