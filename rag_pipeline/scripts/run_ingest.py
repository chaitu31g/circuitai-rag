"""End-to-end ingest script for the CircuitAI RAG pipeline.

Pipeline:
    1. Load all *_chunks.json from rag_chunks/ (or custom path)
    2. Embed with BGEM3Embedder (bge-small-en-v1.5)
    3. Upsert into ChromaDB (data/vectordb/)
    4. Print summary

Usage:
    # From project root (c:/demo):
    python -m rag_pipeline.scripts.run_ingest
    python -m rag_pipeline.scripts.run_ingest --chunks_dir rag_chunks --db_dir data/vectordb
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag_pipeline.embeddings.bge_embedder import BGEM3Embedder
from rag_pipeline.embeddings.embed_pipeline import EmbeddingPipeline
from rag_pipeline.vectordb.chroma_store import ChromaStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CircuitAI RAG ingest pipeline")
    parser.add_argument(
        "--chunks_dir",
        type=Path,
        default=Path("rag_chunks"),
        help="Directory containing *_chunks.json files (default: rag_chunks/)",
    )
    parser.add_argument(
        "--db_dir",
        type=Path,
        default=Path("data/vectordb"),
        help="ChromaDB persistence directory (default: data/vectordb/)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="datasheets",
        help="Chroma collection name (default: datasheets)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(SEPARATOR)
    print("CircuitAI — RAG Ingest Pipeline")
    print(SEPARATOR)

    total_start = time.time()

    # ------------------------------------------------------------------ #
    # Step 1 — Load chunks                                                  #
    # ------------------------------------------------------------------ #
    print(f"\n[1/4] Loading chunks from '{args.chunks_dir}' ...")
    pipeline = EmbeddingPipeline(embedder=BGEM3Embedder())
    raw_chunks = pipeline.load_chunks_from_dir(args.chunks_dir)
    print(f"      Loaded {len(raw_chunks)} raw chunks")

    if not raw_chunks:
        logger.error("No chunks found — aborting")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Step 2 — Embed                                                        #
    # ------------------------------------------------------------------ #
    print(f"\n[2/4] Embedding {len(raw_chunks)} chunks ...")
    embed_start = time.time()
    embedded_chunks = pipeline.run(raw_chunks)
    embed_elapsed = time.time() - embed_start
    dim = len(embedded_chunks[0]["embedding"]) if embedded_chunks else 0
    print(f"      Done — {len(embedded_chunks)} vectors, dim={dim}, "
          f"time={embed_elapsed:.1f}s")

    # ------------------------------------------------------------------ #
    # Step 3 — Store in ChromaDB                                           #
    # ------------------------------------------------------------------ #
    print(f"\n[3/4] Storing in ChromaDB ('{args.db_dir}') ...")
    store = ChromaStore(persist_dir=args.db_dir, collection_name=args.collection)
    store.upsert_chunks(embedded_chunks)
    print(f"      Collection '{args.collection}' now has {store.count} documents")

    # ------------------------------------------------------------------ #
    # Step 4 — Persist (no-op for Chroma >= 0.4, logged for clarity)       #
    # ------------------------------------------------------------------ #
    print("\n[4/4] Persisting ...")
    store.persist()
    print("      Done (auto-persisted)")

    # ------------------------------------------------------------------ #
    # Summary                                                               #
    # ------------------------------------------------------------------ #
    total_elapsed = time.time() - total_start
    print(f"\n{SEPARATOR}")
    print(f"  Chunks ingested : {len(embedded_chunks)}")
    print(f"  Vector dim      : {dim}")
    print(f"  Collection      : {args.collection}")
    print(f"  DB path         : {args.db_dir.resolve()}")
    print(f"  Total time      : {total_elapsed:.1f}s")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
