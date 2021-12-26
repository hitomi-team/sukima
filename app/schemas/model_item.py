from typing import List, Optional

from pydantic import BaseModel


class ModelItemBase(BaseModel):
    model_name: str


class ModelItem(ModelItemBase):
    size: int

    class Config:
        orm_mode = True


class ModelGenArgs(BaseModel):
    max_length: int
    max_time: Optional[float] = None
    min_length: Optional[int] = None
    eos_token_id: Optional[int] = None


class ModelBiasArgs(BaseModel):
    sequence: str
    bias: float
    ensure_sequence_finish: bool
    generate_once: bool


class ModelSampleArgs(BaseModel):
    temp: Optional[float] = None
    top_p: Optional[float] = None
    top_a: Optional[float] = None
    top_k: Optional[int] = None
    tfs: Optional[float] = None
    rep_p: Optional[float] = None
    rep_p_range: Optional[int] = None
    rep_p_slope: Optional[float] = None
    bad_words: List[str] = None
    biases: Optional[List[ModelBiasArgs]] = None


class ModelGenRequest(BaseModel):
    model: str
    prompt: str
    sample_args: ModelSampleArgs
    gen_args: ModelGenArgs


class ModelLoadRequest(BaseModel):
    model: str
    parallel: Optional[bool] = False
    sharded: Optional[bool] = False
