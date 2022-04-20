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
    logprobs: Optional[int] = None
    best_of: Optional[int] = None

class ModelSampleArgs(BaseModel):
    class ModelLogitBiasArgs(BaseModel):
        id: int
        bias: float
    
    class ModelPhraseBiasArgs(BaseModel):
        sequences: List[str]
        bias: float
        ensure_sequence_finish: bool
        generate_once: bool

    temp: Optional[float] = None
    top_p: Optional[float] = None
    top_a: Optional[float] = None
    top_k: Optional[int] = None
    typical_p: Optional[float] = None
    tfs: Optional[float] = None
    rep_p: Optional[float] = None
    rep_p_range: Optional[int] = None
    rep_p_slope: Optional[float] = None
    bad_words: Optional[List[str]] = None
    # logit biases are a list of int and float tuples
    logit_biases: Optional[List[ModelLogitBiasArgs]] = None
    phrase_biases: Optional[List[ModelPhraseBiasArgs]] = None


class ModelGenRequest(BaseModel):
    model: str
    prompt: str
    softprompt: Optional[str] = None
    sample_args: ModelSampleArgs
    gen_args: ModelGenArgs


class ModelClassifyRequest(BaseModel):
    model: str
    prompt: str
    labels: List[str]

class ModelLoadRequest(BaseModel):
    model: str
    parallel: Optional[bool] = False
    sharded: Optional[bool] = False
    quantized: Optional[bool] = False
    tensorize: Optional[bool] = False
    device: Optional[str] = 'cpu'
