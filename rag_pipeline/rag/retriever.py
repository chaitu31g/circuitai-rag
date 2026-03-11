"""Retriever layer for CircuitAI datasheet RAG.

Keeps retrieval logic behind a small API so future reranking or hybrid search
can be inserted without changing callers.

Query-intent routing
─────────────────────
The retriever classifies each query into one of four intents:

  • FEATURE   — "feature", "key features", "advantages", "benefits", "highlights"
  • SPEC      — "specification", "rating", "electrical characteristics", etc.
  • GRAPH     — "graph", "plot", "curve", "chart", "figure"
  • GENERAL   — everything else (pure vector similarity)

For FEATURE and SPEC queries a metadata-filtered primary pass retrieves the
correct section first, then vector similarity fills any remaining budget.

For GRAPH queries, all figure chunks are retrieved via metadata filter and
merged with vector results so none are missed.

Figure-chunk deprioritization
──────────────────────────────
For non-GRAPH queries, figure chunks that sneak into vector results are
penalised (score × 0.7) before sorting so textual sections rank higher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from rag_pipeline.embeddings.bge_embedder import BGEM3Embedder
from rag_pipeline.vectordb.base import VectorStore


# ── Query intent detection ─────────────────────────────────────────────────────

_FEATURE_KEYWORDS = {
    "feature", "features", "key feature", "key features",
    "advantage", "advantages", "benefit", "benefits",
    "highlight", "highlights",
}

_SPEC_KEYWORDS = {
    "specification", "specifications", "spec", "specs",
    "rating", "ratings", "electrical characteristic", "electrical characteristics",
    "absolute maximum", "recommended operating", "thermal characteristic",
    "typical characteristic", "parameter", "parameters",
}

_GRAPH_KEYWORDS = {
    "graph", "graphs", "plot", "plots", "curve", "curves",
    "chart", "charts", "figure", "figures",
    "characteristic", "waveform", "vs", "versus",
}

# Section names stored in ChromaDB metadata for each intent
_INTENT_SECTION_MAP = {
    "feature": "features",
    "spec":    "electrical_characteristics",
    "graph":   "figures_and_diagrams",
}

# Score multiplier applied to figure chunks for non-graph queries
_FIGURE_SCORE_PENALTY = 0.7


def _classify_intent(query: str) -> str:
    """Classify query into 'feature', 'spec', 'graph', or 'general'."""
    q = query.lower()
    if any(kw in q for kw in _FEATURE_KEYWORDS):
        return "feature"
    if any(kw in q for kw in _SPEC_KEYWORDS):
        return "spec"
    if any(kw in q for kw in _GRAPH_KEYWORDS):
        return "graph"
    return "general"


def is_graph_query(query: str) -> bool:
    """Return True if the query is asking about graphs, charts, or characteristic curves."""
    return _classify_intent(query) == "graph"


# ── Retriever config ───────────────────────────────────────────────────────────

@dataclass
class RetrieverConfig:
    """Settings for vector retrieval."""

    top_k: int = 25


class QueryEmbedder(Protocol):
    """Minimal embedder contract used by Retriever."""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        ...


# ── Main retriever ─────────────────────────────────────────────────────────────

class Retriever:
    """Query embedder + vector store adapter with query-intent routing."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Optional[QueryEmbedder] = None,
        config: Optional[RetrieverConfig] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder or BGEM3Embedder()
        self.config = config or RetrieverConfig()

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-k chunks for a user query with intent-aware routing.

        Routing behaviour
        -----------------
        FEATURE / SPEC queries:
          1. Metadata-filtered pass → retrieves the dedicated section first.
          2. Vector similarity pass → fills remaining slots.
          Figure chunks from vector pass are penalised (×0.7) and sorted to
          the back so the correct section always leads.

        GRAPH queries:
          1. Vector similarity pass.
          2. Metadata-filtered pass → injects ALL figure chunks so none are
             missed by embedding similarity alone.

        GENERAL queries:
          Standard vector similarity pass only.  Figure chunks that
          appear in the results are penalised (×0.7) and re-sorted.
        """
        query = query.strip()
        if not query:
            return []

        k = top_k or self.config.top_k
        intent = _classify_intent(query)
        query_embedding = self.embedder.embed_texts([query])[0]
        normalized_filters = self._normalize_filters(filters)

        if intent == "graph":
            results = self._retrieve_graph(query_embedding, k, normalized_filters)
        elif intent in ("feature", "spec"):
            results = self._retrieve_section_first(
                query_embedding, k, normalized_filters, intent
            )
        else:
            # General query — pure vector pass + figure penalty
            results = self.vector_store.query(
                query_embedding=query_embedding,
                n_results=k,
                filters=normalized_filters,
            )
            results = self._penalise_figures(results)
            results.sort(key=lambda d: d.get("score", 0.0), reverse=True)

        return results

    # ── Intent-specific helpers ────────────────────────────────────────────────

    def _retrieve_section_first(
        self,
        query_embedding: List[float],
        k: int,
        normalized_filters: Optional[Dict[str, Any]],
        intent: str,
    ) -> List[Dict[str, Any]]:
        """Fetch the target section first via metadata filter, then fill with vector results."""
        collection = getattr(self.vector_store, "collection", None)
        section_name = _INTENT_SECTION_MAP[intent]
        section_docs: List[Dict[str, Any]] = []

        if collection is not None:
            try:
                where: dict = {"section_name": {"$eq": section_name}}
                if normalized_filters and "part_number" in normalized_filters:
                    where = {
                        "$and": [
                            {"part_number": {"$eq": normalized_filters["part_number"]}},
                            {"section_name": {"$eq": section_name}},
                        ]
                    }
                raw = collection.get(
                    where=where,
                    include=["documents", "metadatas", "ids"],
                )
                ids   = raw.get("ids",       []) or []
                docs  = raw.get("documents", []) or []
                metas = raw.get("metadatas") or [{}] * len(ids)

                for did, dtxt, dmeta in zip(ids, docs, metas):
                    section_docs.append({
                        "id":       did,
                        "text":     dtxt,
                        "metadata": dmeta,
                        # Boost section-matched chunks so they always lead
                        "score":    1.5,
                    })
            except Exception:
                pass  # non-fatal: fall through to vector-only path

        # Vector similarity pass to fill remaining budget
        seen_ids: set = {d.get("id") for d in section_docs}
        remaining = max(k - len(section_docs), k)  # always do a full pass for ranking
        vector_results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=remaining,
            filters=normalized_filters,
        )

        # Merge: add vector results not already in the section-filtered set
        for vdoc in vector_results:
            if vdoc.get("id") not in seen_ids:
                # Penalise figure chunks that sneak into non-graph results
                if self._is_figure(vdoc):
                    vdoc = dict(vdoc)
                    vdoc["score"] = vdoc.get("score", 0.0) * _FIGURE_SCORE_PENALTY
                section_docs.append(vdoc)
                seen_ids.add(vdoc.get("id"))

        # Sort: section-filtered chunks lead (score=1.5), then by similarity
        section_docs.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        return section_docs[:k]

    def _retrieve_graph(
        self,
        query_embedding: List[float],
        k: int,
        normalized_filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Vector pass + inject all figure chunks for graph-related queries."""
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=k,
            filters=normalized_filters,
        )

        collection = getattr(self.vector_store, "collection", None)
        if collection is not None:
            try:
                where: dict = {"$or": [
                    {"type":       {"$eq": "figure"}},
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
                pass  # non-fatal: fall back to vector results only

        return results

    # ── Utility helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _is_figure(doc: Dict[str, Any]) -> bool:
        meta = doc.get("metadata") or {}
        return (
            meta.get("chunk_type") == "figure"
            or meta.get("type") == "figure"
            or meta.get("section_name") == "figures_and_diagrams"
        )

    @staticmethod
    def _penalise_figures(
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Reduce scores of figure chunks so textual sections rank higher."""
        penalised = []
        for doc in results:
            meta = doc.get("metadata") or {}
            if (
                meta.get("chunk_type") == "figure"
                or meta.get("type") == "figure"
                or meta.get("section_name") == "figures_and_diagrams"
            ):
                doc = dict(doc)
                doc["score"] = doc.get("score", 0.0) * _FIGURE_SCORE_PENALTY
            penalised.append(doc)
        return penalised

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
