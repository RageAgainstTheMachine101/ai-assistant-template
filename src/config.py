import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # OpenAI settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    
    # Vector store settings
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "vector_store")
    
    # Embedding settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # RAG settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # API settings
    max_tokens: int = int(os.getenv("MAX_TOKENS", "500"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))

# Create settings instance
settings = Settings()