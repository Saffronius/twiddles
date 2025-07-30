import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    openai_api_key: str
    embed_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o"  # Upgraded from gpt-4o-mini for better responses
    router_threshold: float = 0.45
    token_budget: int = 8000
    
    class Config:
        env_file = ".env"

settings = Settings() 