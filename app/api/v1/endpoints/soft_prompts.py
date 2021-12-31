import base64
import json

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

import app.crud.soft_prompt as crud
from app.api.deps import get_current_approved_user, get_session, get_current_user
from app.models.user import User
from app.schemas.soft_prompt import SoftPromptCreate

router = APIRouter()

@router.get("/my")
async def get_user_soft_prompts(current_user: User = Depends(get_current_approved_user), session: AsyncSession = Depends(get_session)):
    soft_prompts = await crud.soft_prompt.get_by_creator(session, creator=current_user)
    return [sp.asdict() for sp in soft_prompts]

@router.post("/upload")
async def upload_soft_prompt(file: UploadFile = File(...), current_user: User = Depends(get_current_approved_user), session: AsyncSession = Depends(get_session)): # noqa
    try:
        contents = json.load(file.file)
        metadata = SoftPromptCreate(**contents)
        data = base64.b64decode(contents["data"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Malformed soft prompt JSON\n{e}")

    try:
        db_obj = await crud.soft_prompt.upload_soft_prompt(session, creator=current_user, data=data, obj_in=metadata)
    except LookupError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return db_obj.asdict()

@router.get("/{id}")
async def get_soft_prompt(id: str, current_user: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)):
    db_obj = await crud.soft_prompt.get(session, id)

    if db_obj is None:
        raise HTTPException(status_code=404, detail="Soft prompt not found.")

    if not db_obj.public and current_user.id != db_obj.creator:
        raise HTTPException(status_code=403, detail="You are not authorized to view this soft prompt.")

    return db_obj.asdict()

@router.delete("/{id}")
async def delete_soft_prompt(id: str, current_user: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)): # noqa
    db_obj = await crud.soft_prompt.get(session, id)

    if db_obj is not None:
        if current_user.id != db_obj.creator:
            raise HTTPException(status_code=403, detail="You are not allowed to delete this soft prompt.")

        await crud.soft_prompt.remove(session, id=id)

    return {"message": "Deleted soft prompt successfully."}
