from typing import List, Optional

import app.core.db.crud as crud
from app.core.db import schemas, models
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.sukima.gpthf import GPTHF
from app.sukima.util import Util
from app.sukima.models import gpt_models

router = APIRouter(dependencies=[Depends(crud.get_db)])


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


@router.post("/users/register", response_model=schemas.User)
async def create_user(user: schemas.UserCreate, db: Session = Depends(crud.get_db)):
    db_user = crud.get_user(db, user.username)

    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered!!!")

    return crud.create_user(db, user=user)


# TODO: I need to replace util.py someday soon.
@router.get("/models")
async def get_model_list():
    model_dict = {"models": {}}

    for model in gpt_models:
        model_dict["models"][model.model_name] = {"ready": True}

    return model_dict


@router.post("/load")
async def load_model(request: ModelLoadRequest, current_user: models.User = Depends(crud.get_current_approved_user)):
    if not current_user.approved:
        raise HTTPException(status_code=401)
    # Check that model exists
    if gpt_models is not None:
        for m in gpt_models:
            if m.model_name == request.model:
                return Util.error(None, "Model already loaded")
    try:
        model = GPTHF(model_name=request.model, parallelize=request.parallel, sharded=request.sharded)
        gpt_models.append(model)
        return Util.success("Loaded model")
    except Exception:
        return Util.error(None, "Unsupported model")


@router.post("/generate")
async def generate(request: ModelGenRequest):
    for m in gpt_models:
        if m.model_name == request.model:
            try:
                return Util.completion(m.generate(request.dict()))
            except Exception as e:
                return Util.error(None, f'Invalid request body! \n{e}')

    return Util.error(None, "Model not found")
