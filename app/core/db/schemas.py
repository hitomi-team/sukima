from typing import List

from pydantic import BaseModel

# Pydantic models go here (schemas)


class ModelItemBase(BaseModel):
    model_name: str


class ModelItemCreate(ModelItemBase):
    pass


class ModelItem(ModelItemBase):
    size: int

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    username: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    approved: bool  # This is not implemented
    allowed_models: List[ModelItem] = []

    class Config:
        orm_mode = True
