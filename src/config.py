#config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from loguru import logger
import sys


class Settings(BaseSettings):
    """Application configuration using Pydantic Settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Environment
    environment: str = "development"
    log_level: str = "INFO"
    
    # PostgreSQL Database Settings (for user data and history)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "health_companion"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    
    # PostgreSQL Vector Store Settings (for knowledge base with embeddings)
    postgres_vector_host: str = "localhost"
    postgres_vector_port: int = 5432
    postgres_vector_db: str = "health_companion_vectors"
    postgres_vector_user: str = "postgres"
    postgres_vector_password: str = "postgres"
    
    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.7
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    
    # LLM Provider Selection
    llm_provider: str = "openai"  # or "ollama"
    
    # Agent Configuration
    max_agent_iterations: int = 10
    agent_verbose: bool = True
    
    # Application Settings
    default_user_id: int = 1
    knowledge_base_dir: str = "data/knowledge_base"
    
    @property
    def database_url(self) -> str:
        """Generate PostgreSQL database URL for user data."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            f"?client_encoding=utf8"
        )
    
    @property
    def postgres_vector_url(self) -> str:
        """Generate PostgreSQL database URL for vector store."""
        return (
            f"postgresql://{self.postgres_vector_user}:{self.postgres_vector_password}"
            f"@{self.postgres_vector_host}:{self.postgres_vector_port}/{self.postgres_vector_db}"
            f"?client_encoding=utf8"
        )
    
    def setup_logging(self) -> None:
        """Configure loguru logger."""
        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr,
            level=self.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        )
        logger.add(
            "logs/arc_health_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="7 days",
            level=self.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
        )


# Global settings instance
settings = Settings()
settings.setup_logging()