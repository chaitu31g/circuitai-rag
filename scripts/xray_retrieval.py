"""scripts/xray_retrieval.py
──────────────────────────────────────────────────────────────────────────────
X-Ray debugging script for the CircuitAI RAG pipeline.

Run in Colab:
    !cd /content/drive/MyDrive/circuitai-rag && \\
        PYTHONPATH=. python scripts/xray_retrieval.py \\
        --query "Extract ALL rows for Continuous drain current" \\
        --component IRF540N \\
        --top-k 60

This script prints EXACTLY what context was handed to the LLM, so you can
confirm once and for all whether the 70°C row is present in the retrieved
chunks or is being dropped by the context assembler.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

# ── Bootstrap PYTHONPATH ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import config
from rag_pipeline.embeddings.bge_embedder import BGEM3Embedder
from rag_pipeline.rag.retriever import Retriever, RetrieverConfig
from rag_pipeline.rag.rag_pipeline import RAGPipeline, RAGConfig
from rag_pipeline.vectordb.chroma_store import ChromaStore
from rag_pipeline.models.qwen_llm import build_prompt, generate_response


SEP  = "─" * 72
SEP2 = "═" * 72


def _truncate(text: str, limit: int = 600) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n  … [{len(text) - limit} more chars]"


def xray(query: str, component_filter: str | None, top_k: int) -> None:

    print(f"\n{SEP2}")
    print("  CircuitAI X-Ray Retrieval Debugger")
    print(SEP2)
    print(f"  Query     : {query}")
    print(f"  Component : {component_filter or '(all)'}")
    print(f"  top_k     : {top_k}")
    print(SEP2 + "\n")

    # ── 1. Retrieval ──────────────────────────────────────────────────────────
    store = ChromaStore(
        persist_dir=Path(config.chroma_persist_dir),
        collection_name=config.chroma_collection,
    )
    embedder = BGEM3Embedder()
    retriever = Retriever(
        vector_store=store,
        embedder=embedder,
        config=RetrieverConfig(top_k=top_k),
    )

    part_filter = {"part_number": component_filter} if component_filter else None
    retrieved = retriever.retrieve(query=query, top_k=top_k, filters=part_filter)

    print(f"[STEP 1] Vector DB returned {len(retrieved)} chunks")
    print(SEP)

    # ── 2. Print every retrieved chunk ────────────────────────────────────────
    target_lower = query.lower()

    for i, doc in enumerate(retrieved, 1):
        text     = doc.get("text", "")
        score    = doc.get("score", 0.0)
        meta     = doc.get("metadata") or {}
        chunk_id = doc.get("id", "?")

        # Flag chunks that contain the parameter we're hunting for
        hit_25  = "25" in text
        hit_70  = "70" in text
        hit_150 = "150" in text
        has_target = any(kw in text.lower() for kw in ["drain current", "continuous"])

        flag = ""
        if has_target:
            flag = "  ◀ PARAMETER MATCH"
            if hit_25:  flag += " [25°C]"
            if hit_70:  flag += " [70°C ← LOOKING FOR THIS]"
            if hit_150: flag += " [150°C]"

        print(f"Chunk #{i:02d}  score={score:.4f}  id={chunk_id}{flag}")
        print(f"  section : {meta.get('section_name') or meta.get('section', '?')}")
        print(f"  type    : {meta.get('chunk_type') or meta.get('type', '?')}")
        print(f"  page    : {meta.get('page', '?')}")
        print(f"  text    →\n{textwrap.indent(_truncate(text), '    ')}")
        print(SEP)

    # ── 3. Context assembly (mirrors app.py logic exactly) ────────────────────
    print(f"\n[STEP 2] Context Assembly (max_context_chars={config.__dict__.get('max_context_chars', 12000)})")
    print(SEP)

    pipeline = RAGPipeline(
        retriever=retriever,
        config=RAGConfig(
            top_k=top_k,
            max_context_chars=12000,
            default_trimmed_chunks=999,   # ← disable auto-trim for X-ray
        ),
    )
    assembled = pipeline.assemble_context(retrieved, debug=True)

    used_docs    = assembled["used_docs"]
    context      = assembled["context"]
    trimmed      = assembled["trimmed_count"]
    debug_events = assembled.get("debug_events", [])

    print(f"  Retrieved chunks : {len(retrieved)}")
    print(f"  Chunks USED      : {len(used_docs)}")
    print(f"  Chunks TRIMMED   : {trimmed}  ← if >0, rows are being DROPPED before LLM sees them")
    if debug_events:
        for evt in debug_events:
            print(f"  ⚠ DEBUG EVENT    : {evt}")
    print()

    # Check if the 70°C row survived into the final context
    context_lower = context.lower()
    for temp_label, present in [
        ("25°C",  "25" in context),
        ("70°C",  "70" in context),
        ("150°C", "150" in context),
    ]:
        status = "✅ PRESENT" if present else "❌ MISSING — DROPPED before LLM"
        print(f"  {temp_label} in final context : {status}")

    print(f"\n[STEP 3] Full context sent to LLM ({len(context)} chars)")
    print(SEP)
    print(context)
    print(SEP)

    # ── 4. Generate and print LLM answer ──────────────────────────────────────
    print(f"\n[STEP 4] LLM Generation")
    print(SEP)
    messages = build_prompt(context=context, query=query)
    try:
        answer = generate_response(
            messages,
            model_id=config.hf_model,
            max_new_tokens=900,
            temperature=0.1,
        )
        print(answer)
    except Exception as exc:
        print(f"  LLM unavailable (model not loaded): {exc}")
        print("  → Context printed above is what WOULD be sent to the LLM.")
    print(SEP)

    # ── 5. Diagnosis summary ──────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  X-Ray DIAGNOSIS")
    print(SEP2)

    has_70c_in_retrieved = any("70" in (d.get("text") or "") for d in retrieved)
    has_70c_in_context   = "70" in context

    if not has_70c_in_retrieved:
        print("  ❌ ROOT CAUSE: The 70°C row chunk does NOT EXIST in ChromaDB.")
        print("     Fix: Re-ingest the PDF after deleting chroma_db/.")
        print("     The ffill fix in parameter_extractor.py will create this chunk.")
    elif not has_70c_in_context:
        print("  ❌ ROOT CAUSE: The 70°C chunk IS in ChromaDB but was TRIMMED by assemble_context().")
        print("     Fix: Increase max_context_chars or disable default_trimmed_chunks auto-trim.")
        print("     See the hardened config below.")
    else:
        print("  ⚠  The 70°C row IS reaching the LLM context.")
        print("     Root cause is PROMPT/LLM failure, not retrieval.")
        print("     The few-shot + CoT prompts should address this.")

    print(SEP2 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CircuitAI X-Ray Retrieval Debugger")
    parser.add_argument("--query", default="Extract ALL rows for Continuous drain current")
    parser.add_argument("--component", default=None, help="Part number filter, e.g. IRF540N")
    parser.add_argument("--top-k", type=int, default=60)
    args = parser.parse_args()

    xray(query=args.query, component_filter=args.component, top_k=args.top_k)
