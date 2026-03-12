import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_pipeline.rag.retriever import Retriever, classify_query_type, detect_query_sections
from rag_pipeline.vectordb.base import VectorStore


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeStore(VectorStore):
    def __init__(self):
        self.last_query = None

    def upsert_chunks(self, chunks):
        return None

    def query(self, query_embedding, n_results=5, filters=None):
        self.last_query = {
            "query_embedding": query_embedding,
            "n_results": n_results,
            "filters": filters,
        }
        return [
            {
                "id": "doc-1",
                "text": "Parameter: Input capacitance",
                "metadata": {"type": "table_row", "section_name": "electrical_characteristics"},
                "score": 0.92,
            }
        ]

    def persist(self):
        return None


def _contains_clause(where, expected):
    if where == expected:
        return True
    if isinstance(where, dict):
        for key in ("$and", "$or"):
            values = where.get(key)
            if isinstance(values, list) and any(_contains_clause(value, expected) for value in values):
                return True
    return False


def test_query_classification_examples():
    assert classify_query_type("Dynamic characteristics") == "table_query"
    assert classify_query_type("Typical transfer characteristics graph") == "graph_query"
    assert classify_query_type("What is the drain current") == "general_query"


def test_dynamic_characteristics_maps_to_section_filter():
    assert "electrical_characteristics" in detect_query_sections("Dynamic characteristics")


def test_table_query_uses_type_and_section_filters():
    store = FakeStore()
    retriever = Retriever(vector_store=store, embedder=FakeEmbedder())

    retriever.retrieve(
        query="Dynamic characteristics",
        top_k=7,
        filters={"part_number": "BSS138N"},
    )

    where = store.last_query["filters"]

    assert _contains_clause(where, {"part_number": {"$eq": "BSS138N"}})
    assert _contains_clause(where, {"type": {"$eq": "table_row"}})
    assert _contains_clause(where, {"chunk_type": {"$eq": "parameter_row"}})
    assert _contains_clause(where, {"section_name": {"$eq": "electrical_characteristics"}})


def test_general_query_keeps_user_filters_without_table_or_graph_type_clause():
    store = FakeStore()
    retriever = Retriever(vector_store=store, embedder=FakeEmbedder())

    retriever.retrieve(
        query="What is the drain current",
        top_k=5,
        filters={"part_number": "BSS138N"},
    )

    where = store.last_query["filters"]

    assert where == {"part_number": {"$eq": "BSS138N"}}
