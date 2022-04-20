import time

import traceback

import app.crud.soft_prompt as crud
from app.api.deps import get_current_approved_user, get_session
from app.gpt.berthf import BERTHF
from app.gpt.gpthf import GPTHF
from app.gpt.models import gpt_models
from app.gpt.utils import is_decoder
from app.schemas.model_item import ModelGenRequest, ModelLoadRequest, ModelClassifyRequest, ModelHiddenRequest
from app.schemas.user import User

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from transformers import AutoConfig

router = APIRouter()


@router.get("/")
async def get_model_list():
    model_dict = {"models": {}}

    for model in gpt_models:
        model_dict["models"][model.model_name] = {"ready": True}

    return model_dict


@router.post("/load")
async def load_model(request: ModelLoadRequest, current_user: User = Depends(get_current_approved_user)): # noqa
    # Check that model exists
    if gpt_models is not None:
        for m in gpt_models:
            if m.model_name == request.model:
                raise HTTPException(status_code=400, detail="Model already loaded")

    try:
        if is_decoder(AutoConfig.from_pretrained(request.model)):
            model = GPTHF(model_name=request.model, device=request.device, parallelize=request.parallel, sharded=request.sharded, quantized=request.quantized, tensorized=request.tensorize)
        else:
            model = BERTHF(model_name=request.model, device=request.device, parallelize=request.parallel, sharded=request.sharded, quantized=request.quantized, tensorized=request.tensorize)

        gpt_models.append(model)

        return {"message": f"Successfully loaded model: {request.model}"}

    except Exception as e:
        return HTTPException(status_code=400, detail=f"Unable to load the model!\n{e}\n{traceback.format_exc()}")


@router.post("/generate")
async def generate(request: ModelGenRequest, current_user: User = Depends(get_current_approved_user), session: AsyncSession = Depends(get_session)): # noqa
    for m in gpt_models:
        if m.model_name == request.model:
            db_softprompt = None
            if request.softprompt:
                db_softprompt = await crud.soft_prompt.get(session, request.softprompt)
                if db_softprompt is None:
                    raise HTTPException(status_code=400, detail=f"No soft prompt with UUID {request.softprompt} exists!") # noqa
            try:
                if not m.decoder:
                    raise RuntimeError("This is not a decoder model!")
                return m.generate(request.dict(), db_softprompt=db_softprompt)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Unable to generate!\n{e}\n{traceback.format_exc()}")

    raise HTTPException(status_code=404, detail="Model not found.")

@router.post("/classify")
async def classify(request: ModelClassifyRequest, current_user: User = Depends(get_current_approved_user)): # noqa
    for m in gpt_models:
        if m.model_name == request.model:
            try:
                return m.classify(request.dict())
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid request body!\n{e}")
        
    raise HTTPException(status_code=404, detail="Model not found.")

@router.post("/hidden")
async def hidden(request: ModelHiddenRequest, current_user: User = Depends(get_current_approved_user)): # noqa
    for m in gpt_models:
        if m.model_name == request.model:
            try:
                return m.hidden(request.dict())
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid request body!\n{e}")
    
    raise HTTPException(status_code=404, detail="Model not found.")
