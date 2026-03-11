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
    generate_response,
    stream_response,
)

# build_synthesis_prompt was added in a later revision of qwen_llm.py.
# Import it gracefully so that a stale Colab Drive copy of qwen_llm.py
# (missing this function) does not prevent the server from starting.
try:
    from rag_pipeline.models.qwen_llm import build_synthesis_prompt  # noqa: F401
except ImportError:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "build_synthesis_prompt not found in rag_pipeline.models.qwen_llm "
        "(stale file?). Using built-in fallback definition."
    )

    _SYNTHESIS_SYSTEM = (
        "You are an expert electronics engineer analyzing a semiconductor datasheet. "
        "You will receive section-level summaries covering different aspects of the component. "
        "Your task is to synthesize the information ACROSS all sections to provide a direct, concise answer. "
        "Preserve all numeric values, units, and conditions accurately. "
        "If a value is absent from all sections, say so clearly."
    )

    def build_synthesis_prompt(section_context: str, query: str) -> list[dict]:  # noqa: F811
        """Fallback definition used when qwen_llm.py on Colab Drive is stale."""
        user_content = (
            "Below are section-level summaries of a semiconductor datasheet.\n"
            "Each section covers a different aspect of the component.\n\n"
            "Synthesize the relevant information from these sections into a complete, direct answer. "
            "Cite specific values, units, and conditions from the summaries.\n\n"
            f"Datasheet Section Summaries:\n{section_context}\n\n"
            f"Question:\n{query}"
        )
        return [
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            {"role": "user",   "content": user_content},
        ]
