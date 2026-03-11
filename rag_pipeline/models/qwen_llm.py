"""
rag_pipeline/models/qwen_llm.py
────────────────────────────────────────────────────────────────────────────
Qwen3.5-4B primary reasoning module for CircuitAI RAG.

This is the canonical LLM service used for answering datasheet questions.
It wraps the same singleton + thread-safe loading pattern used throughout
the codebase, and exposes a simple generate_response() function that the
backend chat endpoints can call directly.

Model   : Qwen/Qwen3.5-4B  (instruction-tuned, supports chat template)
Quant   : 4-bit NF4 via bitsandbytes  — fits comfortably on a T4 (16 GB)
Device  : device_map="auto"  (GPU if available, CPU fallback)

The model can also be overridden by the HF_MODEL environment variable so
Colab notebooks can pin to a specific checkpoint without code changes.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Generator

logger = logging.getLogger(__name__)

# ── Model identifier ───────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("HF_MODEL", "Qwen/Qwen3.5-4B")

# ── Module-level singletons ────────────────────────────────────────────────────
_tokenizer = None
_model     = None
_load_lock = threading.Lock()

# System prompt injected via the chat template for every query.
_SYSTEM_PROMPT = (
    "You are an expert electronics engineer answering questions based solely on the provided datasheet context. "
    "Provide your answer immediately and directly. "
    "Never invent, estimate, or infer values not present in the context. "
    "If the answer cannot be found in the provided context, say so clearly."
)

# Reasoning-step patterns to strip from model output.
_REASONING_PATTERNS = (
    r"(?im)^\s*\*?\*?(?:step\s*\d+|analyze the request|review the context?"
    r"|thinking process|analysis|extract features|identify relevant)"
    r"[^\n]*\n?"
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(model_id: str = MODEL_NAME) -> None:
    """Load tokenizer + 4-bit quantised Qwen3.5-4B into module globals (once)."""
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
                "  pip install transformers accelerate bitsandbytes sentencepiece safetensors"
            ) from exc

        logger.info("Loading Qwen3.5-4B: %s …", model_id)

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
            torch_dtype="auto",
        )
        _model.eval()
        logger.info("Qwen3.5-4B loaded ✓  (model=%s, device_map=auto)", model_id)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_model_once(model_id: str = MODEL_NAME) -> None:
    """Pre-load the model at startup so the first chat request is fast."""
    _load_model(model_id)


def build_prompt(context: str, query: str) -> list[dict]:
    """Build the chat messages list for a strict answer-only RAG query.

    Returns a list of message dicts ({'role': ..., 'content': ...}) which are
    passed to the tokenizer's apply_chat_template(). This approach works
    correctly for Qwen3.5 and any other instruction-tuned model without
    manual prompt format management.

    The prompt is structured to produce a concise final answer with no
    chain-of-thought reasoning or step-by-step analysis output.
    """
    user_content = (
        "You are an expert electronics engineer.\n\n"
        "Provide a concise, direct technical explanation (2-4 sentences) answering the user's question. "
        "Base your entire answer strictly on the datasheet context below.\n\n"
        f"Datasheet Context:\n{context}\n\n"
        f"Question:\n{query}"
    )
    return [
        {"role": "system",  "content": _SYSTEM_PROMPT},
        {"role": "user",    "content": user_content},
    ]


def build_synthesis_prompt(section_context: str, query: str) -> list[dict]:
    """Build a chat messages list for multi-section reasoning (section-summarization pipeline).

    Unlike ``build_prompt()``, the context here is a structured block of
    pre-summarized datasheet sections (produced by ``SectionSummarizer``),
    separated by ``## Section Name`` headings.

    The model is instructed to reason *holistically* across all sections
    rather than extracting from any single one.
    """
    system = (
        "You are an expert electronics engineer analyzing a semiconductor datasheet. "
        "You will receive section-level summaries covering different aspects of the component. "
        "Your task is to synthesize the information ACROSS all sections to provide a direct, concise answer. "
        "Preserve all numeric values, units, and conditions accurately. "
        "If a value is absent from all sections, say so clearly."
    )
    user_content = (
        "Below are section-level summaries of a semiconductor datasheet.\n"
        "Each section covers a different aspect of the component.\n\n"
        "Synthesize the relevant information from these sections into a complete, direct answer. "
        "Cite specific values, units, and conditions from the summaries.\n\n"
        f"Datasheet Section Summaries:\n{section_context}\n\n"
        f"Question:\n{query}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
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


def _filter_reasoning_steps(text: str) -> str:
    """Remove chain-of-thought reasoning lines that Qwen3.5-4B may still emit.

    Strips lines whose heading matches common reasoning-step patterns such as:
    "Analyze the Request", "Review the Chunks", "Step 1", "Thinking Process".
    Only the heading line (and any immediately following indented sub-bullet
    lines) is removed; the conclusive answer sentences are preserved.
    """
    import re

    # Phase 1 – remove numbered / bulleted reasoning headers and their body.
    # Matches lines like: "1. Analyze the Request", "**Step 2**", "## Analysis"
    heading_re = re.compile(
        r"(?im)"
        r"^[ \t]*(?:[\*#\-]+[ \t]*)?"      # optional markdown prefix
        r"(?:\d+\.[ \t]*)?"                 # optional numeric list prefix
        r"(?:\*\*)?"                         # optional bold open
        r"(?:"
        r"analyze the request"
        r"|review the chunks?"
        r"|thinking process"
        r"|step\s*\d+"
        r"|analysis"
        r"|extract features"
        r"|identify relevant"
        r")"
        r"(?:\*\*)?"                         # optional bold close
        r"[^\n]*\n?"                         # rest of line
    )
    cleaned = heading_re.sub("", text)

    # Phase 2 – collapse multiple blank lines left by removal.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def generate_response(
    prompt: list[dict] | str,
    model_id: str = MODEL_NAME,
    max_new_tokens: int = 300,
    temperature: float = 0.4,
    top_p: float = 0.9,
    do_sample: bool = True,
    repetition_penalty: float = 1.1,
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
        Max tokens to generate. 300 is sufficient for synthesized answers.
    temperature:
        Sampling temperature. 0.4 balances creativity and faithfulness,
        encouraging synthesis rather than verbatim chunk copying.
    top_p:
        Nucleus sampling probability mass. 0.9 allows flexible language
        generation while avoiding low-probability token noise.
    do_sample:
        True = sampling; False = greedy decoding.
    repetition_penalty:
        Values > 1.0 penalise repeated tokens, reducing chunk repetition.

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
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=_tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_length:]
    raw = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return _filter_reasoning_steps(raw)


def stream_response(
    prompt: list[dict] | str,
    model_id: str = MODEL_NAME,
    max_new_tokens: int = 300,
    temperature: float = 0.4,
    top_p: float = 0.9,
    do_sample: bool = True,
    repetition_penalty: float = 1.1,
) -> Generator[str, None, None]:
    """Stream tokens from Qwen3.5-4B one by one.

    Uses transformers.TextIteratorStreamer to push tokens from a background
    thread while the calling generator yields them to the SSE endpoint.

    Parameters
    ----------
    prompt:
        Either the list of chat messages (from build_prompt()) or a raw string.
    max_new_tokens:
        Max tokens to stream. Matches generate_response() default.
    temperature:
        0.4 encourages synthesis over verbatim extraction.
    top_p:
        Nucleus sampling mass — 0.9 keeps output fluent and diverse.
    do_sample:
        True = sampling; prevents strict greedy chunk copying.
    repetition_penalty:
        > 1.0 discourages repetition of chunk phrases.

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
        "streamer":           streamer,
        "max_new_tokens":     max_new_tokens,
        "temperature":        temperature,
        "top_p":              top_p,
        "do_sample":          do_sample,
        "repetition_penalty": repetition_penalty,
        "pad_token_id":       _tokenizer.eos_token_id,
    }

    gen_thread = threading.Thread(
        target=lambda: _model.generate(**gen_kwargs),
        daemon=True,
        name="qwen-generate",
    )
    gen_thread.start()

    for token in streamer:
        if token:
            yield token

    gen_thread.join()
