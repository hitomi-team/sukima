from typing import List

from app.schemas.model_item import ModelItem
from pydantic import BaseModel


class UserBase(BaseModel):
    username: str
    email: str
    permission_level: int


class UserCreate(UserBase):
    password: str


class UserUpdate(UserBase):
    # TODO: fill this in
    pass


class User(UserBase):
    permission_level: int
    allowed_models: List[ModelItem] = []

    class Config:
        orm_mode = True
