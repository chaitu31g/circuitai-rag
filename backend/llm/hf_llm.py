"""
backend/llm/hf_llm.py
─────────────────────
HuggingFace Transformers-based LLM module for CircuitAI RAG.

Replaces the Ollama integration so the backend runs inside Google Colab
(or any environment with a CUDA-capable GPU) without needing a sidecar
service.

Model  : mistralai/Mistral-7B-Instruct-v0.2
Quant  : 4-bit (bitsandbytes NF4) – fits comfortably on a T4 (16 GB VRAM)
Device : auto-selected by `device_map="auto"` (GPU if available, else CPU)
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ── Module-level singletons (loaded once at startup) ──────────────────────────
_tokenizer = None
_model = None
_load_lock = threading.Lock()

# Default model – override via the HF_MODEL env variable or config
DEFAULT_MODEL = os.environ.get(
    "HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"
)


def _load_model(model_id: str = DEFAULT_MODEL) -> None:
    """Load the tokenizer and 4-bit quantized model into the module globals.

    Calling this more than once is a no-op (guarded by a threading lock).
    """
    global _tokenizer, _model

    with _load_lock:
        if _model is not None:
            return  # Already loaded

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Required packages are missing. Install them with:\n"
                "  pip install transformers accelerate bitsandbytes"
            ) from exc

        logger.info("Loading HuggingFace model: %s …", model_id)

        # 4-bit NF4 quantisation config – keeps VRAM usage manageable on T4
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",          # GPU if CUDA available, else CPU
            torch_dtype=torch.float16,
        )
        _model.eval()
        logger.info("Model loaded successfully ✓  (device_map=auto)")


def load_model_once(model_id: str = DEFAULT_MODEL) -> None:
    """Public entry-point called at FastAPI startup to pre-load the model."""
    _load_model(model_id)


def generate_response(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    do_sample: bool = True,
    model_id: str = DEFAULT_MODEL,
) -> str:
    """Generate a text response for *prompt* using the loaded HF model.

    Parameters
    ----------
    prompt:
        The full prompt string (already includes system context + user query).
    max_new_tokens:
        Maximum number of tokens the model may generate.
    temperature:
        Sampling temperature. Lower = more deterministic answers.
    do_sample:
        Whether to use sampling (True) or greedy decoding (False).
    model_id:
        Model to load if not already loaded (default: Mistral-7B-Instruct-v0.2).

    Returns
    -------
    str
        The decoded generated text (prompt stripped from output).
    """
    # Lazy-load on first call if startup didn't pre-load
    if _model is None:
        _load_model(model_id)

    import torch

    inputs = _tokenizer(prompt, return_tensors="pt")

    # Move input tensors to the same device as the model's first parameter
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (strip the prompt)
    input_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_length:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_prompt(context: str, query: str) -> str:
    """Build the standardised RAG prompt for the electronics datasheet assistant.

    Parameters
    ----------
    context:
        The retrieved and assembled context chunks from ChromaDB.
    query:
        The user's question.

    Returns
    -------
    str
        The formatted prompt string ready for :func:`generate_response`.
    """
    return (
        "You are an expert electronics datasheet assistant.\n\n"
        "Use the provided datasheet context to answer the question accurately.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer clearly and technically."
    )
