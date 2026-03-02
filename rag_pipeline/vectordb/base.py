"""Abstract base for vector store backends.

Provides a swappable interface so ChromaStore, QdrantStore, or MilvusStore
can all be used without changing calling code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorStore(ABC):
    """Minimal contract every vector store backend must fulfil."""

    @abstractmethod
    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Insert or update chunks in the store.

        Each chunk must contain:
            id         : str  — unique identifier
            text       : str  — original chunk text (stored as document)
            embedding  : list[float]
            metadata   : dict — arbitrary key/value pairs for filtering
        """

    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Nearest-neighbour search.

        Args:
            query_embedding: Dense query vector.
            n_results: Maximum number of results to return.
            filters: Optional metadata equality filters (backend-specific format).

        Returns:
            List of result dicts with at least 'id', 'text', 'metadata', 'score'.
        """

    @abstractmethod
    def persist(self) -> None:
        """Flush any in-memory state to disk (no-op for auto-persisting backends)."""
