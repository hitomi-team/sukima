import os.path
from typing import Optional, Dict, Any

from pydantic import BaseSettings, validator, PostgresDsn


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "e846cf2582aa9af7102158500b6b60d88f1c89f6de07e2b9ecd5013c4217adb0"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_SERVER: str = "postgres"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "sukimadb"

    class Config:
        env_file = "conf.env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

    def get_url(self):
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@" \
               f"{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"


settings = Settings()
