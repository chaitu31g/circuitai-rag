"""Embedding pipeline — loads chunk JSON, attaches embeddings and IDs.

Takes the flat chunk JSON produced by ingestion/chunker.py and enriches
each chunk with:
  - a deterministic ID (sha256 of text+componentId, first 16 chars)
  - a dense embedding vector from BGEM3Embedder
  - a 'metadata' wrapper containing all non-text fields

Output schema per chunk:
{
    "id":        str,          # deterministic, safe for upsert
    "text":      str,          # original chunk text — never modified
    "embedding": list[float],  # from BGEM3Embedder
    "metadata":  dict          # componentId, chunkType, + any extras
}

Design: this class only transforms data — it never writes to disk or DB.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from rag_pipeline.embeddings.bge_embedder import BGEM3Embedder

logger = logging.getLogger(__name__)


def _make_id(text: str, component_id: str) -> str:
    """Deterministic 16-char hex ID — idempotent across re-runs."""
    raw = f"{component_id}::{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class EmbeddingPipeline:
    """Orchestrates loading, embedding, and repackaging of chunk dicts.

    Args:
        embedder: A BGEM3Embedder instance. Created automatically if None.
    """

    def __init__(self, embedder: BGEM3Embedder | None = None) -> None:
        self.embedder = embedder or BGEM3Embedder()

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @staticmethod
    def load_chunks(path: Path) -> List[Dict[str, Any]]:
        """Load a single chunk JSON file and return its list of chunk dicts."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")

        logger.info("Loaded %d chunks from %s", len(data), path.name)
        return data

    @staticmethod
    def load_chunks_from_dir(chunk_dir: Path) -> List[Dict[str, Any]]:
        """Load all *_chunks.json files from a directory."""
        chunk_dir = Path(chunk_dir)
        all_chunks: List[Dict[str, Any]] = []

        json_files = sorted(chunk_dir.glob("*_chunks.json"))
        if not json_files:
            logger.warning("No *_chunks.json files found in %s", chunk_dir)
            return all_chunks

        for json_path in json_files:
            all_chunks.extend(EmbeddingPipeline.load_chunks(json_path))

        logger.info("Total chunks loaded from dir: %d", len(all_chunks))
        return all_chunks

    # ------------------------------------------------------------------
    # Core transform
    # ------------------------------------------------------------------

    def run(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embed chunks and return enriched output dicts.

        Steps:
          1. Filter out chunks with empty text.
          2. Extract texts for batch encoding.
          3. Call embedder once — efficient batching.
          4. Rebuild each chunk with id + embedding + metadata wrapper.

        Args:
            chunks: Raw chunk dicts from load_chunks / load_chunks_from_dir.

        Returns:
            List of enriched chunk dicts ready for ChromaStore.upsert_chunks().
        """
        # Step 1 — filter invalid chunks
        valid: List[Dict[str, Any]] = [c for c in chunks if c.get("text", "").strip()]
        skipped = len(chunks) - len(valid)
        if skipped:
            logger.warning("Skipped %d chunks with empty text", skipped)

        if not valid:
            logger.error("No valid chunks to embed")
            return []

        # Step 2 — collect texts
        texts = [c["text"].strip() for c in valid]

        # Step 3 — batch embed (single model.encode() call)
        embeddings = self.embedder.embed_texts(texts)

        # Step 4 — repackage as output schema
        results: List[Dict[str, Any]] = []
        for chunk, embedding in zip(valid, embeddings):
            # chunk keys (after asdict): text, chunk_type, metadata:{part_number,...}
            # Flatten so part_number etc. live at the top level of stored metadata
            inner_meta = chunk.get("metadata", {})
            component_id = (
                inner_meta.get("part_number")
                or chunk.get("componentId", "unknown")
            )

            metadata: Dict[str, Any] = {
                "chunk_type": chunk.get("chunk_type") or chunk.get("chunkType", "unknown"),
            }
            metadata.update(inner_meta)   # brings part_number, section_name, etc. up

            results.append({
                "id": _make_id(chunk["text"], component_id),
                "text": chunk["text"],
                "embedding": embedding,
                "metadata": metadata,
            })

        logger.info(
            "Pipeline complete: %d chunks embedded (dim=%d)",
            len(results),
            len(results[0]["embedding"]) if results else 0,
        )
        return results
