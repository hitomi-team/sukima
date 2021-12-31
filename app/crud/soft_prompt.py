from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import uuid4

from app.core.config import settings
from app.crud.base import CrudBase
from app.gpt.models import gpt_models
from app.models.soft_prompt import SoftPrompt
from app.models.user import User
from app.schemas.soft_prompt import SoftPromptCreate, SoftPromptUpdate


class CrudSoftPrompt(CrudBase[SoftPrompt, SoftPromptCreate, SoftPromptUpdate]):
    async def upload_soft_prompt(self, session: AsyncSession, *, creator: User, data: bytes, obj_in: SoftPromptCreate) -> SoftPrompt: # noqa
        # was there supposed to be a database table for this?
        model_exists = False
        for model in gpt_models:
            if model.model_name == obj_in.model:
                model_exists = True
                break
        if not model_exists:
            raise LookupError(f"Model {obj_in.model} has not been loaded.")

        db_obj = SoftPrompt(
            id=str(uuid4()),
            name=obj_in.name,
            description=obj_in.description,
            public=obj_in.public,
            creator=creator.id,
            loss=obj_in.loss,
            steps=obj_in.steps,
            model=obj_in.model
        )

        session.add(db_obj)
        await session.commit()
        await session.refresh(db_obj)

        db_obj.write(data)

        return db_obj

    async def get_by_creator(self, session: AsyncSession, *, creator: User) -> List[SoftPrompt]:
        return (await session.execute(select(self.model).where(self.model.creator == creator.id))).scalars().all()


soft_prompt = CrudSoftPrompt(SoftPrompt)
