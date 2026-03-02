"""RAG inference components for CircuitAI.

This package is intentionally modular:
- Retriever: vector search + metadata filtering
- Prompt builder: datasheet-grounded instruction assembly
- RAG pipeline: orchestration + local Ollama inference
"""

from rag_pipeline.rag.prompt_builder import DatasheetPromptBuilder
from rag_pipeline.rag.rag_pipeline import OllamaClient, RAGConfig, RAGPipeline
from rag_pipeline.rag.reranker import CrossEncoderReranker
from rag_pipeline.rag.retriever import Retriever, RetrieverConfig

__all__ = [
    "CrossEncoderReranker",
    "DatasheetPromptBuilder",
    "OllamaClient",
    "RAGConfig",
    "RAGPipeline",
    "Retriever",
    "RetrieverConfig",
]
