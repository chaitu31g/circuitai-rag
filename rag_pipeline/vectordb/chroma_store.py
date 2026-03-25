from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import chromadb

from rag_pipeline.vectordb.base import VectorStore

logger = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str = "datasheets",
        expected_dim: Optional[int] = None,
    ):
        self.persist_dir       = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name  = collection_name
        self._expected_dim     = expected_dim

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir)
        )

        self._collection = self._get_or_recreate_collection()

        logger.info(
            "ChromaStore ready — collection='%s', path='%s', existing docs=%d",
            collection_name,
            self.persist_dir,
            self.count(),
        )

    def _get_or_recreate_collection(self):
        """Return the collection, recreating it if the dimension has changed.

        Old bge-small-en-v1.5 collections have dim=384.
        The new bge-m3 collections have dim=1024.
        Dimension mismatch causes silent wrong results or hard crashes,
        so we drop-and-recreate rather than append.
        """
        col = self._client.get_or_create_collection(
            name=self._collection_name
        )

        if self._expected_dim is not None and col.count() > 0:
            # Probe the stored dimension by fetching one embedding
            try:
                sample = col.get(limit=1, include=["embeddings"])
                stored_embs = sample.get("embeddings") or []
                if stored_embs and len(stored_embs[0]) != self._expected_dim:
                    stored_dim = len(stored_embs[0])
                    logger.warning(
                        "ChromaStore: collection '%s' has dim=%d but expected dim=%d. "
                        "Deleting existing collection and rebuilding.",
                        self._collection_name, stored_dim, self._expected_dim,
                    )
                    self._client.delete_collection(self._collection_name)
                    col = self._client.get_or_create_collection(
                        name=self._collection_name
                    )
                    logger.info(
                        "ChromaStore: collection '%s' recreated (dim=%d).",
                        self._collection_name, self._expected_dim,
                    )
            except Exception as exc:
                logger.warning(
                    "ChromaStore: dimension check failed (%s) — proceeding anyway.", exc
                )

        return col

    # -------------------------------------------------
    # UPSERT CHUNKS  (satisfies abstract method)
    # -------------------------------------------------
    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Insert or update a list of enriched chunk dicts.

        Each chunk must have:
            id        : str
            text      : str
            embedding : list[float]
            metadata  : dict
        """
        if not chunks:
            logger.warning("upsert_chunks called with empty list — skipping")
            return

        ids = [c["id"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks into ChromaDB", len(chunks))

    # -------------------------------------------------
    # PERSIST  (satisfies abstract method)
    # -------------------------------------------------
    def persist(self) -> None:
        """No-op — PersistentClient auto-persists after every write."""
        logger.debug("ChromaStore.persist() called — auto-persisted by PersistentClient")

    # -------------------------------------------------
    # QUERY  (satisfies abstract method)
    # -------------------------------------------------
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Nearest-neighbour search. Returns list of result dicts."""
        # ChromaDB raises an error if n_results > collection size.
        # CRITICAL: if metadata tags (where={...}) are used, n_results MUST 
        # not exceed the number of documents that match the filter. Otherwise,
        # hnswlib crashes with 'RuntimeError: Cannot return results as 2D array'.
        total = self._collection.count()
        if total == 0:
            return []

        effective_n = n_results
        if filters:
            try:
                # Get the IDs of all documents matching the metadata filter
                res = self._collection.get(where=filters, include=[])
                filtered_count = len(res.get("ids", []))
                effective_n = min(n_results, filtered_count)
            except Exception as exc:
                logger.warning("ChromaStore: filtered count failed (%s)", exc)
                effective_n = min(n_results, total)
        else:
            effective_n = min(n_results, total)

        if effective_n <= 0:
            logger.debug("ChromaStore: zero results match filters — skipping query")
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_n,
            where=filters,
        )

        # Flatten ChromaDB's batched response into a clean list
        output = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_id, text, meta, dist in zip(ids, docs, metas, distances):
            output.append({
                "id": doc_id,
                "text": text,
                "metadata": meta,
                "score": 1.0 - dist,  # convert distance → similarity
            })

        return output

    # -------------------------------------------------
    # DELETE
    # -------------------------------------------------
    def delete(self, ids: Iterable[str]) -> None:
        self._collection.delete(ids=list(ids))

    def delete_component(self, part_number: str) -> int:
        """Delete all chunks for a specific part number. Returns count of deleted docs."""
        # Use get to find all IDs for this part_number first
        res = self._collection.get(where={"part_number": part_number}, include=[])
        ids = res.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.info("Deleted %d chunks for component '%s'", len(ids), part_number)
        return len(ids)

    # -------------------------------------------------
    # COUNT
    # -------------------------------------------------
    def count(self) -> int:
        return self._collection.count()

    # -------------------------------------------------
    # LIBRARY — unique components in the collection
    # -------------------------------------------------
    def get_library(self) -> List[Dict[str, Any]]:
        """Return one entry per unique component (part_number) with stats.

        Each entry:
            {
              "component_id": str,
              "chunk_count":  int,
              "chunk_types":  list[str],   # distinct chunk_type values
            }
        """
        total = self.count()
        if total == 0:
            return []

        # Fetch all metadata in batches (ChromaDB has no GROUP BY)
        result = self._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []

        by_component: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            part = meta.get("part_number") or meta.get("componentId") or "unknown"
            if part not in by_component:
                by_component[part] = {"chunk_count": 0, "chunk_types": set()}
            by_component[part]["chunk_count"] += 1
            chunk_type = meta.get("chunkType") or meta.get("chunk_type") or "text"
            by_component[part]["chunk_types"].add(chunk_type)

        return [
            {
                "component_id": k,
                "chunk_count":  v["chunk_count"],
                "chunk_types":  sorted(v["chunk_types"]),
            }
            for k, v in sorted(by_component.items())
        ]

    # -------------------------------------------------
    # RAW COLLECTION ACCESS (Optional helper)
    # -------------------------------------------------
    @property
    def collection(self):
        return self._collection