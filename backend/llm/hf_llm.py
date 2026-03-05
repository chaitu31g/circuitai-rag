"""
backend/llm/hf_llm.py
─────────────────────
HuggingFace Transformers LLM module for CircuitAI RAG.

Designed for Google Colab (CUDA GPU) or CPU-only environments.
Supports any instruction-tuned causal LM on HuggingFace Hub.

Model  : set via HF_MODEL env var (default: Qwen/Qwen2.5-3B-Instruct)
Quant  : 4-bit NF4 (bitsandbytes) — fits on a T4 (16 GB VRAM)
Device : auto-selected by device_map="auto"

Prompt format is handled automatically by the tokenizer's apply_chat_template(),
so switching models (Qwen, Mistral, Llama, Gemma, etc.) never requires prompt
format changes.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Generator

logger = logging.getLogger(__name__)

# ── Module-level singletons ────────────────────────────────────────────────────
_tokenizer = None
_model     = None
_load_lock = threading.Lock()

DEFAULT_MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# System prompt injected into the chat template for every query.
_SYSTEM_PROMPT = (
    "You are an expert electronics datasheet assistant. "
    "Use ONLY the provided datasheet context to answer the question accurately. "
    "If the answer is not present in the context, say so clearly. "
    "Do not invent values, specifications, or conditions."
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(model_id: str = DEFAULT_MODEL) -> None:
    """Load tokenizer + 4-bit quantised model into module globals (once)."""
    global _tokenizer, _model

    with _load_lock:
        if _model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "Required packages missing. Install with:\n"
                "  pip install transformers accelerate bitsandbytes"
            ) from exc

        logger.info("Loading HuggingFace model: %s …", model_id)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        _tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Ensure a pad token exists (some models omit it)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        _model.eval()
        logger.info("Model loaded ✓  (model=%s, device_map=auto)", model_id)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_model_once(model_id: str = DEFAULT_MODEL) -> None:
    """Pre-load the model at startup so the first chat request is fast."""
    _load_model(model_id)


def build_prompt(context: str, query: str) -> list[dict]:
    """Build the chat messages list for the RAG query.

    Returns a list of message dicts ({'role': ..., 'content': ...}) which are
    passed to the tokenizer's apply_chat_template(). This approach works
    correctly for ANY instruction-tuned model (Qwen, Mistral, Llama, Gemma…)
    without manual prompt format management.
    """
    user_content = (
        f"Datasheet Context:\n{context}\n\n"
        f"Question:\n{query}"
    )
    return [
        {"role": "system",  "content": _SYSTEM_PROMPT},
        {"role": "user",    "content": user_content},
    ]


def _apply_template(messages: list[dict]) -> str:
    """Convert chat messages → model-specific prompt string via the tokenizer."""
    if _tokenizer is None:
        raise RuntimeError("Model is not loaded yet.")
    return _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # adds the assistant turn opener
    )


def generate_response(
    prompt: list[dict] | str,
    model_id: str = DEFAULT_MODEL,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    do_sample: bool = True,
) -> str:
    """Generate a complete response (non-streaming).

    Parameters
    ----------
    prompt:
        Either the list of chat messages returned by build_prompt(), or
        a raw string (legacy path — still supported).
    model_id:
        HuggingFace model repo ID. Used only if model isn't loaded yet.
    max_new_tokens:
        Max tokens to generate.
    temperature:
        Sampling temperature — lower is more deterministic.
    do_sample:
        True = sampling; False = greedy decoding.

    Returns
    -------
    str
        The generated text with the prompt stripped out.
    """
    if _model is None:
        _load_model(model_id)

    import torch

    # Accept both messages list and raw string
    text = _apply_template(prompt) if isinstance(prompt, list) else prompt

    inputs = _tokenizer(text, return_tensors="pt")
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

    input_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_length:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def stream_response(
    prompt: list[dict] | str,
    model_id: str = DEFAULT_MODEL,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    do_sample: bool = True,
) -> Generator[str, None, None]:
    """Stream tokens from the HuggingFace model one by one.

    Uses transformers.TextIteratorStreamer to push tokens from a background
    thread while the calling generator yields them to the SSE endpoint.

    Parameters
    ----------
    prompt:
        Either the list of chat messages (from build_prompt()) or a raw string.

    Yields
    ------
    str
        Each token string as it is decoded.
    """
    if _model is None:
        _load_model(model_id)

    import torch
    from transformers import TextIteratorStreamer

    text = _apply_template(prompt) if isinstance(prompt, list) else prompt

    inputs = _tokenizer(text, return_tensors="pt")
    device = next(_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        _tokenizer,
        skip_prompt=True,          # don't echo the input prompt
        skip_special_tokens=True,
    )

    gen_kwargs = {
        **inputs,
        "streamer":         streamer,
        "max_new_tokens":   max_new_tokens,
        "temperature":      temperature,
        "do_sample":        do_sample,
        "pad_token_id":     _tokenizer.eos_token_id,
    }

    gen_thread = threading.Thread(
        target=lambda: _model.generate(**gen_kwargs),
        daemon=True,
        name="hf-generate",
    )
    gen_thread.start()

    for token in streamer:
        if token:
            yield token

    gen_thread.join()
