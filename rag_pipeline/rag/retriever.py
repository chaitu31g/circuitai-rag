"""Retriever layer for CircuitAI datasheet RAG.

Keeps retrieval logic behind a small API so future reranking or hybrid search
can be inserted without changing callers.

Graph-aware routing
───────────────────
When the user query contains graph-related keywords (graph, plot, curve, etc.)
the retriever automatically filters results to prioritise figure chunks.
For explicit graph queries, all figure chunks are returned via metadata filter
rather than relying on embedding similarity alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from rag_pipeline.embeddings.bge_embedder import BGEM3Embedder
from rag_pipeline.vectordb.base import VectorStore


# ── Graph query detection ──────────────────────────────────────────────────────
_GRAPH_QUERY_KEYWORDS = {
    "graph", "graphs", "plot", "plots", "curve", "curves",
    "chart", "charts", "figure", "figures",
    "characteristic", "waveform", "vs", "versus",
}


def is_graph_query(query: str) -> bool:
    """Return True if the query is asking about graphs, charts, or characteristic curves."""
    q = query.lower()
    return any(kw in q for kw in _GRAPH_QUERY_KEYWORDS)


@dataclass
class RetrieverConfig:
    """Settings for vector retrieval."""

    top_k: int = 25


class QueryEmbedder(Protocol):
    """Minimal embedder contract used by Retriever."""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        ...


class Retriever:
    """Query embedder + vector store adapter with graph-aware routing."""

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

        For graph-related queries, also injects a metadata-filtered pass that
        retrieves all figure chunks so none are missed by embedding similarity.

        Results are returned in similarity order from the vector store and
        preserve chunk atomicity by operating on stored chunk-level documents.
        """
        query = query.strip()
        if not query:
            return []

        k = top_k or self.config.top_k
        query_embedding = self.embedder.embed_texts([query])[0]
        normalized_filters = self._normalize_filters(filters)

        # Standard vector similarity pass
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=k,
            filters=normalized_filters,
        )

        # Graph-aware supplemental retrieval via metadata filter.
        # When the query is graph-related, fetch ALL figure chunks and merge
        # them with the vector results so no graph is missed.
        if is_graph_query(query):
            try:
                collection = getattr(self.vector_store, "collection", None)
                if collection is not None:
                    where: dict = {"$or": [
                        {"type": {"$eq": "figure"}},
                        {"chunk_type": {"$eq": "figure"}},
                    ]}
                    if normalized_filters and "part_number" in normalized_filters:
                        where = {
                            "$and": [
                                {"part_number": {"$eq": normalized_filters["part_number"]}},
                                where,
                            ]
                        }
                    raw = collection.get(
                        where=where,
                        include=["documents", "metadatas", "ids"],
                    )
                    ids   = raw.get("ids",       []) or []
                    docs  = raw.get("documents", []) or []
                    metas = raw.get("metadatas") or [{}] * len(ids)

                    seen_ids = {d.get("id") for d in results}
                    for did, dtxt, dmeta in zip(ids, docs, metas):
                        if did not in seen_ids:
                            results.append({
                                "id":       did,
                                "text":     dtxt,
                                "metadata": dmeta,
                                "score":    1.0,
                            })
                            seen_ids.add(did)
            except Exception:
                pass   # non-fatal: fall back to vector results only

        return results

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
