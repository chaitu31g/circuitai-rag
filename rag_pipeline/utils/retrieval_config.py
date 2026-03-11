"""
rag_pipeline/utils/retrieval_config.py
──────────────────────────────────────────────────────────────────────────────
Centralised retrieval depth configuration for the CircuitAI RAG system.

Why this module exists
──────────────────────
Previously the retrieval depth was scattered across:
  • RetrieverConfig(top_k=…) calls inside backend/app.py
  • The ChatRequest.top_k default in app.py
  • Ad-hoc literals in scripts and test files

This module defines **one authoritative set of constants** so that tuning
retrieval depth, context budget, and reranker settings requires changing
a single file — not hunting through the whole codebase.

All constants are overridable via environment variables so Colab notebooks
and deployment configs can adjust behaviour without code changes:

    export CIRCUITAI_RETRIEVAL_TOP_K=60
    export CIRCUITAI_RERANK_TOP_N=25

Design constraints
──────────────────
• Does NOT modify any existing function or class.
• Does NOT change function signatures.
• Provides factory helpers (build_retriever_config, build_retriever) that
  wrap the existing RetrieverConfig / Retriever constructors.
• Fully backward-compatible: existing call sites continue to work unchanged.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core retrieval depth
# ─────────────────────────────────────────────────────────────────────────────

# Number of candidate chunks fetched from ChromaDB per query.
#
# Why 40?
# -------
# A typical semiconductor datasheet contains 6-10 distinct sections
# (features, absolute maximum ratings, electrical characteristics, thermal
# characteristics, pin description, application notes, …).  Each section
# may produce 3-8 chunks after dual-pass chunking.  Retrieving 40 candidates
# ensures that even queries touching multiple sections return enough context
# for the LLM to reason holistically, while staying within the 12 000-char
# context budget enforced downstream.
#
# The existing RetrieverConfig already defaults to 50, which is even more
# conservative.  Setting DEFAULT_RETRIEVAL_TOP_K = 40 here leaves headroom
# for the reranker to select the 20 best chunks without starving any section.
#
# Override via environment:
#   export CIRCUITAI_RETRIEVAL_TOP_K=60
DEFAULT_RETRIEVAL_TOP_K: int = int(
    os.environ.get("CIRCUITAI_RETRIEVAL_TOP_K", "40")
)

# ─────────────────────────────────────────────────────────────────────────────
# Reranker budget
# ─────────────────────────────────────────────────────────────────────────────

# After fetching DEFAULT_RETRIEVAL_TOP_K candidates, the CrossEncoderReranker
# selects the best RERANK_TOP_N chunks before section summarization.
#
# Override via environment:
#   export CIRCUITAI_RERANK_TOP_N=25
RERANK_TOP_N: int = int(
    os.environ.get("CIRCUITAI_RERANK_TOP_N", "20")
)

# ─────────────────────────────────────────────────────────────────────────────
# Context window budget
# ─────────────────────────────────────────────────────────────────────────────

# Maximum characters of assembled context text passed to the LLM.
# 12 000 chars ≈ 3 000 tokens — fits within Qwen3.5-4B's effective window
# without forcing truncation of individual chunks.
#
# Override via environment:
#   export CIRCUITAI_MAX_CONTEXT_CHARS=16000
MAX_CONTEXT_CHARS: int = int(
    os.environ.get("CIRCUITAI_MAX_CONTEXT_CHARS", "12000")
)

# Maximum number of chunks passed to assemble_context() for graph queries.
# Graph queries inject ALL figure chunks so a separate hard cap is needed.
#
# Override via environment:
#   export CIRCUITAI_GRAPH_MAX_CHUNKS=25
GRAPH_MAX_CONTEXT_CHUNKS: int = int(
    os.environ.get("CIRCUITAI_GRAPH_MAX_CHUNKS", "20")
)

# ─────────────────────────────────────────────────────────────────────────────
# Section-summarization LLM settings
# ─────────────────────────────────────────────────────────────────────────────

# Max tokens generated per section summary (used by SectionSummarizer).
SECTION_SUMMARY_MAX_TOKENS: int = int(
    os.environ.get("CIRCUITAI_SECTION_SUMMARY_MAX_TOKENS", "200")
)

# Temperature for section summaries — low to stay faithful to source data.
SECTION_SUMMARY_TEMPERATURE: float = float(
    os.environ.get("CIRCUITAI_SECTION_SUMMARY_TEMPERATURE", "0.3")
)

# ─────────────────────────────────────────────────────────────────────────────
# Minimum quality thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Chunks shorter than this (chars) are filtered out before embedding/storage.
MIN_CHUNK_CHARS: int = int(
    os.environ.get("CIRCUITAI_MIN_CHUNK_CHARS", "10")
)

# Figure chunks shorter than this are discarded during ingestion.
MIN_FIGURE_CHUNK_CHARS: int = int(
    os.environ.get("CIRCUITAI_MIN_FIGURE_CHUNK_CHARS", "20")
)


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers  (do NOT modify existing classes — only wrap them)
# ─────────────────────────────────────────────────────────────────────────────

def build_retriever_config(top_k: Optional[int] = None):
    """Return a RetrieverConfig pre-filled with the configured retrieval depth.

    This is a thin factory — it does not change RetrieverConfig in any way.
    Existing code that constructs RetrieverConfig directly continues to work
    unchanged; this helper is for new call sites that want to pick up
    the central configuration automatically.

    Parameters
    ----------
    top_k:
        Override the default retrieval depth for this call.  If ``None``,
        ``DEFAULT_RETRIEVAL_TOP_K`` is used.

    Returns
    -------
    RetrieverConfig
        Ready to pass to ``Retriever(config=…)``.

    Example
    -------
    ::

        from rag_pipeline.utils.retrieval_config import build_retriever_config
        from rag_pipeline.rag.retriever import Retriever

        retriever = Retriever(
            vector_store=store,
            embedder=embedder,
            config=build_retriever_config(),   # top_k=40 from config
        )
    """
    from rag_pipeline.rag.retriever import RetrieverConfig  # local import avoids circulars

    effective_k = top_k if top_k is not None else DEFAULT_RETRIEVAL_TOP_K
    logger.debug("build_retriever_config: top_k=%d", effective_k)
    return RetrieverConfig(top_k=effective_k)


def build_retriever(
    vector_store: Any,
    embedder: Any = None,
    top_k: Optional[int] = None,
    extra_config: Optional[Dict[str, Any]] = None,
):
    """Construct a Retriever wired to the central retrieval configuration.

    This factory wraps the existing ``Retriever.__init__`` without altering
    its signature.  It is the recommended way to obtain a Retriever in new
    code so that changing retrieval depth only requires updating this module.

    Parameters
    ----------
    vector_store:
        A ``VectorStore`` instance (e.g. ``ChromaStore``).
    embedder:
        An embedder instance.  If ``None``, ``BGEM3Embedder`` is used
        (same default as ``Retriever.__init__``).
    top_k:
        Override retrieval depth.  Defaults to ``DEFAULT_RETRIEVAL_TOP_K``.
    extra_config:
        Reserved for future ``RetrieverConfig`` fields.  Currently unused.

    Returns
    -------
    Retriever
        Ready to call ``.retrieve()``.

    Example
    -------
    ::

        from rag_pipeline.utils.retrieval_config import build_retriever

        retriever = build_retriever(vector_store=store, embedder=embedder)
        results   = retriever.retrieve(query="What is the max drain current?")
    """
    from rag_pipeline.rag.retriever import Retriever  # local import avoids circulars

    config = build_retriever_config(top_k=top_k)
    logger.info(
        "build_retriever: top_k=%d  (DEFAULT_RETRIEVAL_TOP_K=%d)",
        config.top_k,
        DEFAULT_RETRIEVAL_TOP_K,
    )
    return Retriever(
        vector_store=vector_store,
        embedder=embedder,
        config=config,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic helper
# ─────────────────────────────────────────────────────────────────────────────

def print_config() -> None:
    """Print the active retrieval configuration to stdout (for debugging)."""
    print("=" * 60)
    print("CircuitAI — Active Retrieval Configuration")
    print("=" * 60)
    print(f"  DEFAULT_RETRIEVAL_TOP_K     : {DEFAULT_RETRIEVAL_TOP_K}")
    print(f"  RERANK_TOP_N                : {RERANK_TOP_N}")
    print(f"  MAX_CONTEXT_CHARS           : {MAX_CONTEXT_CHARS}")
    print(f"  GRAPH_MAX_CONTEXT_CHUNKS    : {GRAPH_MAX_CONTEXT_CHUNKS}")
    print(f"  SECTION_SUMMARY_MAX_TOKENS  : {SECTION_SUMMARY_MAX_TOKENS}")
    print(f"  SECTION_SUMMARY_TEMPERATURE : {SECTION_SUMMARY_TEMPERATURE}")
    print(f"  MIN_CHUNK_CHARS             : {MIN_CHUNK_CHARS}")
    print(f"  MIN_FIGURE_CHUNK_CHARS      : {MIN_FIGURE_CHUNK_CHARS}")
    print("=" * 60)
    print("Override any value via environment variable, e.g.:")
    print("  export CIRCUITAI_RETRIEVAL_TOP_K=60")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
