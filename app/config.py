from pydantic_settings import BaseSettings # <-- Change this line
from pydantic import Field, validator # <-- Keep other imports from pydantic
import logging

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    WEBHOOK_SECRET: str = Field(..., env='WEBHOOK_SECRET')
    DATABASE_URL: str = Field('sqlite:////data/app.db', env='DATABASE_URL')
    LOG_LEVEL: str = Field('INFO', env='LOG_LEVEL')
    API_PORT: int = 8000 # Default port

    @validator('WEBHOOK_SECRET', always=True)
    def check_webhook_secret(cls, v):
        if not v:
            raise ValueError("WEBHOOK_SECRET must be set and non-empty.")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()