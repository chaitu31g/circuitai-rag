"""Retriever layer for CircuitAI datasheet RAG.

Keeps retrieval logic behind a small API so future reranking or hybrid search
can be inserted without changing callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from rag_pipeline.embeddings.bge_embedder import BGEM3Embedder
from rag_pipeline.vectordb.base import VectorStore


@dataclass
class RetrieverConfig:
    """Settings for vector retrieval."""

    top_k: int = 5


class QueryEmbedder(Protocol):
    """Minimal embedder contract used by Retriever."""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        ...


class Retriever:
    """Query embedder + vector store adapter."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Optional[QueryEmbedder] = None,
        config: Optional[RetrieverConfig] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder or BGEM3Embedder()
        self.config = config or RetrieverConfig()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-k chunks for a user query.

        Results are returned in similarity order from the vector store and
        preserve chunk atomicity by operating on stored chunk-level documents.
        """
        query = query.strip()
        if not query:
            return []

        k = top_k or self.config.top_k
        query_embedding = self.embedder.embed_texts([query])[0]
        normalized_filters = self._normalize_filters(filters)
        return self.vector_store.query(
            query_embedding=query_embedding,
            n_results=k,
            filters=normalized_filters,
        )

    @staticmethod
    def _normalize_filters(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Normalize metadata filters for backend compatibility.

        Chroma metadata was persisted as scalar values; this keeps filters
        equality-based and supports CLI patterns like:
            --filter parameter collectorBaseVoltage
        """
        if not filters:
            return None

        cleaned: Dict[str, Any] = {}
        for key, value in filters.items():
            if key is None:
                continue
            if isinstance(value, str):
                cleaned[str(key)] = value.strip()
            else:
                cleaned[str(key)] = value
        return cleaned or None
