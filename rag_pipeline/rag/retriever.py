"""Retriever layer for CircuitAI datasheet RAG.

Keeps retrieval logic behind a small API so future reranking or hybrid search
can be inserted without changing callers.

This retriever now performs query-aware metadata filtering before vector
search. Queries are classified into:

- ``table_query``: retrieve only table rows
- ``graph_query``: retrieve only figure chunks
- ``general_query``: standard semantic retrieval with figure de-prioritization

When the query names a datasheet section such as "dynamic characteristics" or
"absolute maximum ratings", the retriever adds a section filter as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

from rag_pipeline.embeddings.bge_embedder import BGEM3Embedder
from rag_pipeline.vectordb.base import VectorStore


_GRAPH_KEYWORDS = (
    "graph",
    "graphs",
    "plot",
    "plots",
    "curve",
    "curves",
    "chart",
    "charts",
    "figure",
    "figures",
    "waveform",
    "waveforms",
    "safe operating area",
    "soa",
    "transfer characteristic",
    "transfer characteristics",
    "output characteristic",
    "output characteristics",
)

_TABLE_HINT_KEYWORDS = (
    "table",
    "tables",
    "row",
    "rows",
    "parameter",
    "parameters",
    "rating",
    "ratings",
    "specification",
    "specifications",
    "spec",
    "specs",
    "operating conditions",
)

_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "dynamic characteristics": ("electrical_characteristics", "dynamic_characteristics"),
    "electrical characteristics": ("electrical_characteristics",),
    "absolute maximum ratings": ("absolute_maximum_ratings",),
    "maximum ratings": ("absolute_maximum_ratings",),
    "recommended operating conditions": ("recommended_operating_conditions",),
    "operating conditions": ("recommended_operating_conditions",),
    "thermal characteristics": ("thermal_characteristics",),
    "static characteristics": ("electrical_characteristics", "static_characteristics"),
    "switching characteristics": ("electrical_characteristics", "switching_characteristics"),
}

_LEGACY_TYPE_FILTERS: dict[str, tuple[dict[str, dict[str, str]], ...]] = {
    "table_query": (
        {"type": {"$eq": "table_row"}},
        {"chunk_type": {"$eq": "table_row"}},
        {"chunk_type": {"$eq": "parameter_row"}},
    ),
    "graph_query": (
        {"type": {"$eq": "figure"}},
        {"chunk_type": {"$eq": "figure"}},
    ),
}

_FIGURE_SCORE_PENALTY = 0.7


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().split())


def detect_query_sections(query: str) -> List[str]:
    """Return section aliases inferred from the user query."""
    q = _normalize_query(query)
    matches: List[str] = []

    for alias in sorted(_SECTION_ALIASES, key=len, reverse=True):
        if alias in q:
            for section_name in _SECTION_ALIASES[alias]:
                if section_name not in matches:
                    matches.append(section_name)

    return matches


def classify_query_type(query: str) -> str:
    """Classify a query into table_query, graph_query, or general_query."""
    q = _normalize_query(query)
    if not q:
        return "general_query"

    if any(keyword in q for keyword in _GRAPH_KEYWORDS):
        return "graph_query"

    if detect_query_sections(q):
        return "table_query"

    if any(keyword in q for keyword in _TABLE_HINT_KEYWORDS):
        return "table_query"

    return "general_query"


def is_graph_query(query: str) -> bool:
    """Return True when the query is explicitly asking for graph content."""
    return classify_query_type(query) == "graph_query"


@dataclass
class RetrieverConfig:
    """Settings for vector retrieval."""

    top_k: int = 50


class QueryEmbedder(Protocol):
    """Minimal embedder contract used by Retriever."""

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        ...


class Retriever:
    """Query embedder + vector store adapter with metadata-aware retrieval."""

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
        """Return top-k chunks for a user query with query-aware filters."""
        query = query.strip()
        if not query:
            return []

        k = top_k or self.config.top_k
        query_type = classify_query_type(query)
        query_embedding = self.embedder.embed_texts([query])[0]
        normalized_filters = self._normalize_filters(filters)
        combined_filters = self._build_query_filters(
            query=query,
            query_type=query_type,
            filters=normalized_filters,
        )

        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=k,
            filters=combined_filters,
        )

        if query_type == "general_query":
            results = self._penalise_figures(results)
            results.sort(key=lambda d: d.get("score", 0.0), reverse=True)

        return results

    @classmethod
    def _build_query_filters(
        cls,
        query: str,
        query_type: str,
        filters: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        clauses: List[Dict[str, Any]] = []

        for key, value in (filters or {}).items():
            clauses.append({key: {"$eq": value}})

        type_clause = cls._build_type_clause(query_type)
        if type_clause:
            clauses.append(type_clause)

        section_clause = cls._build_section_clause(query, query_type)
        if section_clause:
            clauses.append(section_clause)

        return cls._combine_clauses(clauses)

    @staticmethod
    def _build_type_clause(query_type: str) -> Optional[Dict[str, Any]]:
        type_filters = _LEGACY_TYPE_FILTERS.get(query_type)
        if not type_filters:
            return None
        if len(type_filters) == 1:
            return type_filters[0]
        return {"$or": list(type_filters)}

    @staticmethod
    def _build_section_clause(query: str, query_type: str) -> Optional[Dict[str, Any]]:
        if query_type != "table_query":
            return None

        section_names = detect_query_sections(query)
        if not section_names:
            return None

        section_filters: List[Dict[str, Any]] = []
        for section_name in section_names:
            section_filters.append({"section": {"$eq": section_name}})
            section_filters.append({"section_name": {"$eq": section_name}})

        if len(section_filters) == 1:
            return section_filters[0]
        return {"$or": section_filters}

    @staticmethod
    def _combine_clauses(clauses: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        clean_clauses = [clause for clause in clauses if clause]
        if not clean_clauses:
            return None
        if len(clean_clauses) == 1:
            return clean_clauses[0]
        return {"$and": list(clean_clauses)}

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
            if Retriever._is_figure(doc):
                doc = dict(doc)
                doc["score"] = doc.get("score", 0.0) * _FIGURE_SCORE_PENALTY
            penalised.append(doc)
        return penalised

    @staticmethod
    def _normalize_filters(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Normalize metadata filters for backend compatibility."""
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
