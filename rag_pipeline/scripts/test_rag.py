"""CLI smoke test for end-to-end local CircuitAI RAG inference.

Examples:
    python -m rag_pipeline.scripts.test_rag --query "LED forward voltage"
    python -m rag_pipeline.scripts.test_rag --query "max current" --filter componentId led
    python -m rag_pipeline.scripts.test_rag --query "collector base voltage" --filter parameter collectorBaseVoltage
    python -m rag_pipeline.scripts.test_rag --query "collector base voltage" --mode spec
    python -m rag_pipeline.scripts.test_rag --query "collector base voltage" --mode json_spec
    python -m rag_pipeline.scripts.test_rag --query "collector base voltage" --mode rag_answer
    python -m rag_pipeline.scripts.test_rag --query "collector base voltage" --mode rag_answer --rerank
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag_pipeline.rag.rag_pipeline import RAGConfig, RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CircuitAI local RAG inference test")
    parser.add_argument("--query", type=str, required=True, help="User query")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["qa", "spec", "json_spec", "rag_answer"],
        default="qa",
        help="Prompt mode: qa (default), spec, json_spec, or rag_answer",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Retriever top-k")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking of retrieved chunks",
    )
    parser.add_argument(
        "--rerank_top_n",
        type=int,
        default=None,
        help="Optional cap on reranked chunks kept before context assembly",
    )
    parser.add_argument(
        "--rerank_blend_alpha",
        type=float,
        default=None,
        help="Optional blend weight (0-1): alpha*reranker + (1-alpha)*vector score",
    )
    parser.add_argument(
        "--filter",
        nargs=2,
        metavar=("KEY", "VALUE"),
        default=None,
        help="Metadata filter, e.g. --filter componentId led",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature for Ollama generation",
    )
    parser.add_argument(
        "--max_context_chars",
        type=int,
        default=6000,
        help="Context size cap (characters)",
    )
    parser.add_argument(
        "--max_context_chunks",
        type=int,
        default=None,
        help="Optional hard cap for number of chunks included in prompt context",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Ollama request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--show_metadata",
        action="store_true",
        help="Print metadata for used chunks",
    )
    parser.add_argument(
        "--debug_context",
        action="store_true",
        help="Print assembled context before final answer",
    )
    return parser.parse_args()


def print_docs(label: str, docs: list[Dict[str, Any]], show_metadata: bool) -> None:
    print(f"\n{label} ({len(docs)}):")
    for i, doc in enumerate(docs, 1):
        score = doc.get("score", 0.0)
        doc_id = doc.get("id", "?")
        text = (doc.get("text") or "").replace("\n", " ").strip()
        text_preview = text if len(text) <= 180 else text[:177] + "..."
        print(f"  [{i}] id={doc_id} score={score:.4f}")
        print(f"      {text_preview}")
        if show_metadata:
            print(f"      metadata={doc.get('metadata', {})}")


def main() -> None:
    args = parse_args()
    filters: Optional[Dict[str, str]] = {args.filter[0]: args.filter[1]} if args.filter else None

    rag = RAGPipeline(
        config=RAGConfig(
            top_k=args.top_k,
            temperature=args.temperature,
            max_context_chars=args.max_context_chars,
            request_timeout_s=args.timeout,
        )
    )

    try:
        result = rag.answer(
            query=args.query,
            top_k=args.top_k,
            filters=filters,
            timeout_s=args.timeout,
            max_context_chunks=args.max_context_chunks,
            debug=args.debug_context,
            mode=args.mode,
            use_reranker=args.rerank,
            rerank_top_n=args.rerank_top_n,
            rerank_blend_alpha=args.rerank_blend_alpha,
        )
    except Exception as exc:
        print(f"\n[ERROR] RAG inference failed: {exc}")
        return

    print_docs("Retrieved chunks", result.get("retrieved_docs", []), args.show_metadata)
    print_docs("Context-used chunks", result.get("used_docs", []), args.show_metadata)

    if args.debug_context:
        print("\nDebug metrics:")
        print(f"  trimmed_chunk_count={result.get('trimmed_count', 0)}")
        print(f"  timeout_used_s={result.get('timeout_used_s', args.timeout)}")
        print(f"  prompt_length_chars={result.get('prompt_length_chars', len(result.get('prompt', '')))}")
        events = result.get("debug_events", [])
        if events:
            for event in events:
                print(f"  context_trim_event={event}")
        if args.mode in {"spec", "json_spec", "rag_answer"}:
            print(f"  chunk_ids_used={result.get('chunk_ids_used', [])}")
            print(f"  extracted_numeric_tokens={result.get('extracted_numeric_tokens', [])}")
        if args.rerank:
            print(f"  rerank_error={result.get('rerank_error')}")
            print(f"  original_vector_ranking={result.get('original_vector_ranking', [])}")
            print(f"  reranked_order={result.get('reranked_order', [])}")
            print(f"  reranker_scores={result.get('reranker_scores', [])}")
            print(f"  dropped_after_rerank={result.get('dropped_after_rerank', [])}")
        if args.mode == "json_spec":
            parsed_json = result.get("parsed_json")
            print(f"  parsed_json_object_count={result.get('parsed_json_count', 0)}")
            print("  parsed_json=")
            if parsed_json is None:
                print("    None (model output was not valid JSON)")
            else:
                print(json.dumps(parsed_json, ensure_ascii=False, indent=2))

        print("\nAssembled context:")
        print("-" * 80)
        print(result.get("context", ""))
        print("-" * 80)

    print("\nFinal answer:")
    print(result.get("answer", ""))


if __name__ == "__main__":
    main()
