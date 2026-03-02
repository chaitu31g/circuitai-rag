"""Minimal retrieval smoke test for the CircuitAI RAG pipeline.

Embeds a sample query, queries ChromaDB, and prints the top results.
Run this AFTER run_ingest.py has populated the collection.

Usage:
    # From project root (c:/demo):
    python -m rag_pipeline.scripts.test_retrieval
    python -m rag_pipeline.scripts.test_retrieval --query "LED forward voltage"
    python -m rag_pipeline.scripts.test_retrieval --filter componentId led
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Force UTF-8 output on Windows — chunk text may contain μ, °, ±, etc.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag_pipeline.embeddings.bge_embedder import BGEM3Embedder
from rag_pipeline.vectordb.chroma_store import ChromaStore

# Set info level to see model loading progress
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

SEPARATOR = "=" * 60

# Sample queries that exercise the datasheet knowledge base
DEMO_QUERIES = [
    "What is the forward voltage of an LED?",
    "OR gate maximum propagation delay",
    "zener diode breakdown voltage",
    "PNP transistor collector current",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CircuitAI RAG retrieval test")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom query string (runs demo queries if omitted)",
    )
    parser.add_argument(
        "--n_results",
        type=int,
        default=5,
        help="Number of results to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--filter",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=None,
        help="Metadata filter, e.g. --filter componentId led",
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
    )
    return parser.parse_args()


def run_query(
    query: str,
    embedder: BGEM3Embedder,
    store: ChromaStore,
    n_results: int = 5,
    filters: Optional[dict] = None,
) -> None:
    # Embed the query (single text)
    try:
        query_vec = embedder.embed_texts([query])[0]
    except OSError as e:
        if "paging file is too small" in str(e):
            print("\n[!] ERROR: Windows Virtual Memory (Paging File) is too small to load the model.")
            print("    Fix: Increase your paging file size in Windows Advanced System Settings,")
            print("    or close other memory-heavy applications (like Chrome/Electron).")
        else:
            print(f"\n[!] ERROR loading embedding model: {e}")
        return
    except Exception as e:
        print(f"\n[!] Unexpected error during embedding: {e}")
        return

    # Retrieve from Chroma
    results = store.query(query_vec, n_results=n_results, filters=filters)

    if not results:
        print("  [!] No results returned — is the collection populated?")
        print("      Run:  python -m rag_pipeline.scripts.run_ingest")
        return

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        comp = meta.get("componentId", "?")
        ctype = meta.get("chunkType", "?")
        score = r.get("score", 0.0)
        text = r.get("text", "")
        # Truncate long texts for readability
        display = text if len(text) <= 120 else text[:117] + "..."
        print(f"  [{i}] score={score:.4f} | {comp} / {ctype}")
        print(f"       {display}")


def main() -> None:
    args = parse_args()

    print(SEPARATOR)
    print("CircuitAI — RAG Retrieval Smoke Test")
    print(SEPARATOR)

    # Build shared embedder and store
    embedder = BGEM3Embedder()
    store = ChromaStore(persist_dir=args.db_dir, collection_name=args.collection)

    print(f"\nCollection '{args.collection}' contains {store.count} documents")

    # Build metadata filter if provided
    filters = {args.filter[0]: args.filter[1]} if args.filter else None

    # Single custom query or cycle through demo queries
    queries = [args.query] if args.query else DEMO_QUERIES

    for query in queries:
        print(f"\nQuery: \"{query}\"", flush=True)
        if filters:
            print(f"Filter: {filters}", flush=True)
        print("-" * 50, flush=True)

        print("DEBUG: Calling embedder.embed_texts...", flush=True)
        run_query(query, embedder, store, n_results=args.n_results, filters=filters)
        print("DEBUG: Query complete.", flush=True)

    print(f"\n{SEPARATOR}")
    print("Smoke test complete.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
