from pydantic_settings import BaseSettings
from pathlib import Path

# Compute outside the settings class
PROJECT_ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    # App config
    allow_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # Path config
    upload_dir: str = str(PROJECT_ROOT / "uploads")
    temp_dir: str = str(PROJECT_ROOT / "temp")
    pdfs_dir: str = str(PROJECT_ROOT / "pdfs")
    docling_dir: str = str(PROJECT_ROOT / "docling_output")
    knowledge_dir: str = str(PROJECT_ROOT / "knowledge_json")

    # Ingestion config
    max_file_size_mb: int = 50
    db_dir: str = str(PROJECT_ROOT / "data" / "vectordb")
    collection_name: str = "datasheets"

    # Chroma & AI Model config
    chroma_persist_dir: str = "data/vectordb"
    chroma_collection: str = "datasheets"
    ollama_model: str = "qwen2.5:7b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    chroma_server_nofile: int = 4096

    class Config:
        env_file = ".env"

config = Settings()

# Ensure directories exist
for _dir in [
    config.upload_dir, config.temp_dir, config.db_dir,
    config.pdfs_dir, config.docling_dir, config.knowledge_dir,
]:
    Path(_dir).mkdir(parents=True, exist_ok=True)