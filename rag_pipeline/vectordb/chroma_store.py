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

        from chromadb.config import Settings
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
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
        try:
            res = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters
            )
            
            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            
            all_results = []
            for iid, txt, mta, dst in zip(ids, docs, metas, dists):
                if "component" not in mta: mta["component"] = "unknown"
                all_results.append({
                    "id": iid,
                    "text": txt,
                    "metadata": mta,
                    "score": 1.0 - (dst or 0.0)
                })
            return all_results
        except Exception as e:
            logger.error(f"ChromaStore query failed: {e}")
            return []

    # -------------------------------------------------
    # DELETE
    # -------------------------------------------------
    def delete(self, ids: Iterable[str]) -> None:
        self._collection.delete(ids=list(ids))

    def delete_component(self, part_number: str) -> int:
        """Delete component entries from single collection."""
        try:
            ct = self._collection.count()
            self._collection.delete(where={"component": part_number})
            after = self._collection.count()
            return ct - after
        except: return 0

    # -------------------------------------------------
    # COUNT
    # -------------------------------------------------
    def count(self) -> int:
        return self._collection.count()

    # -------------------------------------------------
    # LIBRARY — unique components in the collection
    # -------------------------------------------------
    def get_library(self) -> List[Dict[str, Any]]:
        raw = self._collection.get(include=["metadatas"])
        metas = raw.get("metadatas") or []
        
        comps = {}
        for m in metas:
            c = m.get("component")
            if c:
                if c not in comps: comps[c] = 0
                comps[c] += 1
                
        result = []
        for c_name, count in comps.items():
            result.append({
                "component_id": c_name,
                "chunk_count": count,
                "chunk_types": ["text", "table", "figure"]
            })
            
        return sorted(result, key=lambda x: x["component_id"])

    # -------------------------------------------------
    # RAW COLLECTION ACCESS (Optional helper)
    # -------------------------------------------------
    @property
    def collection(self):
        return self._collection