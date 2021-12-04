from typing import Optional, Dict, Any

from pydantic import BaseSettings, validator, PostgresDsn
from starlette.config import Config
from starlette.datastructures import Secret


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_SERVER: str = "db"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str
    SQLALCHEMY_DATABASE_URI: str

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True



settings = Settings()