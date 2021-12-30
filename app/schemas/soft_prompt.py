from typing import Optional

from numpy import float64
from pydantic import BaseModel


class SoftPrompt(BaseModel):
    name: str
    description: Optional[str]
    public: Optional[bool]


class SoftPromptCreate(SoftPrompt):
    model: str
    loss: float64
    steps: int


class SoftPromptUpdate(SoftPrompt):
    pass
