from typing import List

from pydantic import BaseModel

from app.schemas.model_item import ModelItem


class UserBase(BaseModel):
    username: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    approved: bool
    allowed_models: List[ModelItem] = []

    class Config:
        orm_mode = True