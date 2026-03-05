from pydantic_settings import BaseSettings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # CORS
    allow_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # Directories
    upload_dir:    str = str(PROJECT_ROOT / "uploads")
    temp_dir:      str = str(PROJECT_ROOT / "temp")
    pdfs_dir:      str = str(PROJECT_ROOT / "pdfs")
    docling_dir:   str = str(PROJECT_ROOT / "docling_output")
    knowledge_dir: str = str(PROJECT_ROOT / "knowledge_json")
    db_dir:        str = str(PROJECT_ROOT / "data" / "vectordb")

    # Ingestion
    max_file_size_mb: int = 50

    # ChromaDB
    chroma_persist_dir: str = "data/vectordb"
    chroma_collection:  str = "datasheets"
    chroma_server_nofile: int = 4096

    # HuggingFace LLM — change HF_MODEL env var in Colab to switch models
    # Examples:
    #   Qwen/Qwen2.5-3B-Instruct   (fast, fits T4 easily)
    #   mistralai/Mistral-7B-Instruct-v0.2  (slower, higher quality)
    #   google/gemma-2b-it           (small & fast)
    hf_model: str = "Qwen/Qwen2.5-3B-Instruct"

    # Embedding & reranker
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model:  str = "BAAI/bge-reranker-base"

    class Config:
        env_file = ".env"


config = Settings()

# Ensure required directories exist
for _dir in [
    config.upload_dir, config.temp_dir, config.db_dir,
    config.pdfs_dir, config.docling_dir, config.knowledge_dir,
]:
    Path(_dir).mkdir(parents=True, exist_ok=True)