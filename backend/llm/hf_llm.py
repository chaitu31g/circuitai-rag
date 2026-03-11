"""
backend/llm/hf_llm.py
─────────────────────
HuggingFace Transformers LLM module for CircuitAI RAG.

Primary reasoning model: Qwen/Qwen3.5-4B
  • Stronger reasoning than Qwen2.5-3B for electronics Q&A
  • Fits on a T4 (16 GB VRAM) with 4-bit NF4 quantisation
  • Device: auto-selected by device_map="auto"

This module is a thin re-export shim that delegates to the canonical
rag_pipeline/models/qwen_llm.py implementation so both the backend
and any scripts importing from this path work identically.

Prompt format is handled automatically by the tokenizer's apply_chat_template(),
so switching models never requires prompt format changes.
"""

from __future__ import annotations

# Re-export the canonical Qwen3.5-4B implementation so existing imports
# from backend.llm.hf_llm continue to work without changes.
from rag_pipeline.models.qwen_llm import (  # noqa: F401
    MODEL_NAME as DEFAULT_MODEL,
    load_model_once,
    build_prompt,
    build_synthesis_prompt,
    generate_response,
    stream_response,
)
