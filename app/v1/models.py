from typing import List, Optional

from pydantic import BaseModel


class ModelItemBase(BaseModel):
    model_name: str


class ModelItem(ModelItemBase):
    size: int

    class Config:
        orm_mode = True


class UserBase(BaseModel):
    username: str


class UserCreate(UserBase):
    password: str


class User(UserBase):
    approved: bool
    allowed_models: List[ModelItem] = []

    class Config:
        orm_mode = True


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


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None