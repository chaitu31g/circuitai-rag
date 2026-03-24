"""
rag_pipeline/models/qwen_llm.py
────────────────────────────────────────────────────────────────────────────
Qwen2.5-3B-Instruct primary reasoning module for CircuitAI RAG.

This is the canonical LLM service used for answering datasheet questions.
It wraps the same singleton + thread-safe loading pattern used throughout
the codebase, and exposes a simple generate_response() function that the
backend chat endpoints can call directly.

Model   : Qwen/Qwen2.5-3B-Instruct  (instruction-tuned, supports chat template)
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


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX → plain-text cleaner
# ─────────────────────────────────────────────────────────────────────────────

def clean_latex_symbols(text: str) -> str:
    """Convert LaTeX math expressions in datasheet text to readable plain text.

    Semiconductor datasheets (and the LLM output derived from them) often
    contain LaTeX-style math notation such as ``$I_{D,pulse}$``.  Since the
    API returns plain text and the UI does not render LaTeX, this function
    converts such expressions into engineering-friendly notation before the
    text is sent to the user.

    Conversions performed
    ---------------------
    * ``$I_{D,pulse}$``  →  ``ID(pulse)``
    * ``$R_{DS(on)}$``   →  ``RDS(on)``
    * ``$T_j$``          →  ``Tj``   (single-char subscript without braces)
    * ``$V^{2}$``        →  ``V^2``
    * ``\\theta``        →  ``theta``
    * ``\\mu``           →  ``µ``
    * ``^\\circ``        →  ``°``
    * ``\\Omega``        →  ``Ω``
    * Removes standalone math delimiters ``$`` and ``\\(`` / ``\\)``.
    """
    import re

    if not text:
        return text

    # ── 1. Remove inline LaTeX fences: \( ... \) and \[ ... \] ────────────
    text = re.sub(r'\\\(', '', text)
    text = re.sub(r'\\\)', '', text)
    text = re.sub(r'\\\[', '', text)
    text = re.sub(r'\\\]', '', text)

    # ── 2. Remove bare $ delimiters (including $$) ─────────────────────────
    text = text.replace('$$', '').replace('$', '')

    # ── 3. Subscripts with braces: X_{abc} → X(abc) ───────────────────────
    #    Handles multi-character subscripts like I_{D,pulse} → ID(pulse)
    text = re.sub(r'([A-Za-z])_\{([^}]*)\}', r'\1(\2)', text)

    # ── 4. Subscripts without braces: X_y → Xy (single char) ─────────────
    text = re.sub(r'([A-Za-z])_([A-Za-z0-9])', r'\1\2', text)

    # ── 5. Superscripts with braces: X^{abc} → X^abc ──────────────────────
    text = re.sub(r'([A-Za-z0-9])\^\{([^}]*)\}', r'\1^\2', text)

    # ── 6. Common engineering / Greek symbols ──────────────────────────────
    replacements = [
        (r'\\theta',   'theta'),
        (r'\\Theta',   'Theta'),
        (r'\\mu',      'µ'),
        (r'\\Omega',   'Ω'),
        (r'\\omega',   'ω'),
        (r'\\alpha',   'alpha'),
        (r'\\beta',    'beta'),
        (r'\\delta',   'delta'),
        (r'\\Delta',   'Delta'),
        (r'\\epsilon', 'epsilon'),
        (r'\\lambda',  'lambda'),
        (r'\\pi',      'pi'),
        (r'\\sigma',   'sigma'),
        (r'\\tau',     'tau'),
        (r'\^\\circ',  '°'),
        (r'\\times',   '×'),
        (r'\\cdot',    '·'),
        (r'\\leq',     '≤'),
        (r'\\geq',     '≥'),
        (r'\\neq',     '≠'),
        (r'\\approx',  '≈'),
        (r'\\infty',   '∞'),
        (r'\\pm',      '±'),
        (r'\\sqrt',    'sqrt'),
        (r'\\frac',    ''),        # remove \frac command; numerator/denominator remain
        (r'\\text\{([^}]*)\}', r'\1'),  # \text{abc} → abc
        (r'\\mathrm\{([^}]*)\}', r'\1'),
        (r'\\mathbf\{([^}]*)\}', r'\1'),
        (r'\\textbf\{([^}]*)\}', r'\1'),
        (r'\\left\(',  '('),
        (r'\\right\)', ')'),
        (r'\\left\[',  '['),
        (r'\\right\]', ']'),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    # ── 7. Strip any remaining lone backslash-word commands ───────────────
    #    e.g. \rm, \bf, \it left over after the above passes
    text = re.sub(r'\\[a-zA-Z]+', '', text)

    return text

# ── Model identifier ───────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# ── Module-level singletons ────────────────────────────────────────────────────
_tokenizer = None
_model     = None
_load_lock = threading.Lock()

# System prompt injected via the chat template for every query.
_SYSTEM_PROMPT = (
    "### ROLE\n"
    "You are a High-Precision Power Electronics Engineer. "
    "Your ONLY job is to extract exact technical specifications from semiconductor datasheets "
    "exactly as they appear — no interpretation, no paraphrasing, no omissions.\n\n"

    "### MANDATORY EXTRACTION PROTOCOL\n"
    "1. SCAN ALL ROWS: Read EVERY row in the provided context before answering. "
    "Do NOT stop at the first matching row.\n"
    "2. CAPTURE ALL CONDITIONS: A parameter (e.g. RDS(on), ID, VGS(th)) may have MULTIPLE rows "
    "for different test conditions (e.g. T=25°C vs T=70°C, VGS=4.5V vs VGS=10V). "
    "You MUST include EVERY such row.\n"
    "3. NEVER COLLAPSE: It is FORBIDDEN to merge multiple rows into a single 'typical' value. "
    "If 3 rows exist for a parameter, output all 3 rows.\n"
    "4. ALWAYS INCLUDE: The 'Test Conditions' column (VGS, ID, Tj, etc.) and Units (V, A, Ω, µA).\n"
    "5. OUTPUT FORMAT: When the answer involves tabular data, format your response "
    "as a Markdown table:\n"
    "   | Parameter | Symbol | Min | Typ | Max | Unit | Conditions |\n"
    "   |-----------|--------|-----|-----|-----|------|------------|\n\n"

    "### STRICT PROHIBITIONS\n"
    "- DO NOT summarize or generalize multiple rows into one line.\n"
    "- DO NOT say 'typical value is X' when the datasheet provides per-condition entries.\n"
    "- DO NOT omit rows due to repeated parameter names — those are separate test conditions.\n"
    "- If data is absent from context, state: "
    "'Data not available in current context.' Do NOT fabricate values.\n"
)

# Reasoning-step patterns to strip from model output.
# Matches blocks starting with "Thinking Process" or similar until we see a final "Answer:" or "Draft X:"
_REASONING_PATTERNS = (
    r"(?im)^\s*\*?\*?(?:step\s*\d+|analyze the request|review the context?"
    r"|thinking process|analysis|extract features|identify relevant)"
    r"[^\n]*\n?"
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(model_id: str = MODEL_NAME) -> None:
    """Load tokenizer + 4-bit quantised Qwen into module globals (once)."""
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

        logger.info("Loading Qwen: %s …", model_id)

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
        logger.info("Qwen loaded ✓  (model=%s, device_map=auto)", model_id)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_model_once(model_id: str = MODEL_NAME) -> None:
    """Pre-load the model at startup so the first chat request is fast."""
    _load_model(model_id)


def build_prompt(context: str, query: str) -> list[dict]:
    """Build the chat messages list for a detailed RAG query.

    Returns a list of message dicts ({'role': ..., 'content': ...}) which are
    passed to the tokenizer's apply_chat_template(). This approach works
    correctly for Qwen3.5 and any other instruction-tuned model without
    manual prompt format management.
    """
    # Clean LaTeX from retrieved context before it enters the prompt so the
    # model never sees raw math notation and is less likely to reproduce it.
    context = clean_latex_symbols(context)

    user_content = (
        f"### CONTEXT:\n{context}\n\n"
        f"### USER QUERY:\n{query}\n\n"
        "### FINAL OUTPUT:\n"
        "(Provide the result as a detailed Markdown table following the datasheet's layout)"
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
        "Your task is to synthesize the information ACROSS all sections and provide a detailed, thorough answer. "
        "Use bullet points (- item) when listing features, specs, or characteristics, and explain each point clearly. "
        "Preserve all numeric values, units, and conditions accurately. "
        "If a value is absent from all sections, say so clearly. "
        "CRITICAL INSTRUCTION: DO NOT output any thinking process, step-by-step analysis, rationale, or drafts. Output EXACTLY AND ONLY the final direct answer."
    )
    # Clean LaTeX from section summaries before sending to the model.
    section_context = clean_latex_symbols(section_context)

    user_content = (
        "Below are section-level summaries of a semiconductor datasheet.\n"
        "Each section covers a different aspect of the component.\n\n"
        "Synthesize the relevant information from these sections into a detailed, well-explained answer.\n"
        "- List each key feature or characteristic as a bullet point.\n"
        "- Explain what each feature means technically (do not just name it).\n"
        "- Cite specific values, units, and conditions from the summaries.\n\n"
        "IMPORTANT: DO NOT output any reasoning steps, 'Thinking Process', or drafts. Output ONLY the final analytical answer.\n\n"
        f"Datasheet Section Summaries:\n{section_context}\n\n"
        f"Question:\n{query}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]



def _apply_template(messages: list[dict]) -> str:
    """Convert chat messages → model-specific prompt string via the tokenizer.

    Passes ``enable_thinking=False`` if supported so reasoning models suppress their 
    built-in chain-of-thought blocks and output the answer directly.
    """
    if _tokenizer is None:
        raise RuntimeError("Model is not loaded yet.")
    try:
        # Some tokenizers support enable_thinking to disable <think> blocks.
        return _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,    # ← disables thinking/reasoning mode if present
        )
    except TypeError:
        # Older tokenizer versions don't have enable_thinking — fall back gracefully.
        return _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _filter_reasoning_steps(text: str) -> str:
    """Remove chain-of-thought / thinking blocks from model output.

    Some models may emit <think>...</think> XML blocks even when
    prompted not to. This function strips them as a safety net, along with
    other common reasoning-step heading patterns.
    """
    import re

    # ── Priority 1: strip <think>...</think> blocks ─────────────────────
    # These are emitted by some model's internal reasoning mode.
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()

    # ── Priority 2: legacy "Thinking Process" / "Draft N:" pattern ────────────
    if re.search(r"(?i)thinking process|analyze the request", text):
        match = re.search(r"(?im)(?:draft\s*\d+|final answer|answer):(.*)", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # ── Priority 3: remove numbered / bulleted reasoning headers ───────────────
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

    # Remove meta-analysis bullet lines like "* Role: Expert Electronics Engineer."
    meta_re = re.compile(
        r"(?im)^[ \t]*\*[ \t]*(?:Role|Task|Constraint|Input Data|Look for):[^\n]*\n?"
    )
    cleaned = meta_re.sub("", cleaned)

    # Collapse multiple blank lines.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def generate_response(
    prompt: list[dict] | str,
    model_id: str = MODEL_NAME,
    max_new_tokens: int = 600,
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
        Max tokens to generate. 600 gives room for detailed bullet-point answers.
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
    raw = clean_latex_symbols(raw)
    return _filter_reasoning_steps(raw)


def stream_response(
    prompt: list[dict] | str,
    model_id: str = MODEL_NAME,
    max_new_tokens: int = 600,
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

    # Buffer all tokens so we can strip <think>...</think> blocks before
    # yielding. Streaming token-by-token with partial think blocks would
    # expose the reasoning text to the client.
    buffer = []
    for token in streamer:
        if token:
            buffer.append(token)

    gen_thread.join()

    full_text = clean_latex_symbols("".join(buffer))
    full_text = _filter_reasoning_steps(full_text)
    # Yield the cleaned text as a single chunk (preserves the streaming interface).
    if full_text:
        yield full_text
