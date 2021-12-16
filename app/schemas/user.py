from typing import List

from app.schemas.model_item import ModelItem
from pydantic import BaseModel


class UserBase(BaseModel):
    username: str
    email: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    permission_level: int
    allowed_models: List[ModelItem] = []

    class Config:
        orm_mode = True
