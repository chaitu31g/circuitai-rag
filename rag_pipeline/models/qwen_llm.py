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
    "You are a High-Precision Power Electronics Engineer and a meticulous data extractor. "
    "Your ONLY job is to extract exact technical specifications from semiconductor "
    "datasheets exactly as they appear — zero interpretation, zero paraphrasing, zero omissions.\n\n"

    "### MANDATORY THINKING PHASE (Chain-of-Thought)\n"
    "Before producing any output, you MUST reason inside a <THINKING> block like this:\n"
    "<THINKING>\n"
    "1. Target Parameter: [Copy the EXACT parameter name from the user query]\n"
    "2. Rows found: [List every row where the Parameter column EXACTLY matches the target, "
    "AND every row where the Parameter cell is BLANK but the nearest non-empty Parameter "
    "cell ABOVE it is an exact match (blank-cell continuation — Rule 0)]\n"
    "3. Rejected rows: [List every row you are discarding and WHY — e.g., "
    "'Gate-source leakage current' — REJECTED: contains 'Gate-source', not 'Drain-source']\n"
    "4. Columns in source: [List the exact column headers from the <DATASHEET_TABLE> tag]\n"
    "</THINKING>\n"
    "This thinking block is mandatory. Do not skip it.\n\n"

    "### RULE 0 — BLANK-CELL CONTINUATION (CRITICAL — PDF TABLE CONVENTION)\n"
    "In PDF datasheets, a parameter with multiple test conditions occupies multiple rows. "
    "ONLY THE FIRST ROW has the parameter name in the Parameter column. "
    "ALL subsequent condition rows for the SAME parameter have a BLANK or EMPTY Parameter cell — "
    "they inherit the parameter name from the nearest non-empty Parameter cell ABOVE them. "
    "YOU MUST include these blank-cell rows in your output, filling the Parameter cell with the inherited name. "
    "REAL EXAMPLE: 'Drain-source leakage current' row 1 → T_j=25°C, value=0.1 µA. "
    "Row 2 → BLANK Parameter cell, T_j=150°C, value=5 µA. BOTH rows are 'Drain-source leakage current'. "
    "The next row with a NEW non-blank Parameter name (e.g. 'Gate-source leakage current') is a DIFFERENT parameter — REJECT IT.\n\n"

    "### RULE 1 — EXACT-MATCH ONLY (NO FUZZY MATCHING)\n"
    "A row matches the query ONLY if: (a) its Parameter cell exactly matches the target name, OR "
    "(b) its Parameter cell is BLANK and the nearest non-empty Parameter cell above it exactly matches. "
    "'Gate-source leakage current' is NOT 'Drain-source leakage current'. "
    "'Gate threshold voltage' is NOT 'Drain threshold voltage'. "
    "Partial word matches are FORBIDDEN.\n\n"

    "### RULE 2 — COLUMN FIDELITY (NO COMPRESSION)\n"
    "You MUST preserve the exact numerical columns from the source table. "
    "If the source has 'min', 'typ', 'max' columns, output ALL THREE as separate columns. "
    "DO NOT compress them into a single 'Value' column. "
    "If a cell is blank or a dash (-), preserve it as-is.\n\n"

    "### RULE 3 — EXHAUSTIVE EXTRACTION\n"
    "After confirming exact matches in the <THINKING> block, output EVERY confirmed row. "
    "Multiple test conditions (T=25°C, T=150°C, VGS=4.5V, VGS=10V) = multiple table rows. "
    "Merging condition rows is FORBIDDEN.\n\n"

    "### RULE 4 — OUTPUT FORMAT\n"
    "After the </THINKING> block, output ONLY a Markdown table. "
    "Use the EXACT column headers from the <DATASHEET_TABLE> context. "
    "No prose, no explanation, no summary before or after the table.\n\n"

    "### STRICT PROHIBITIONS\n"
    "- DO NOT output rows whose Parameter does not exactly match the target "
    "(e.g. if target is 'Drain-source leakage current', do NOT output 'Gate-source leakage current').\n"
    "- DO NOT skip blank-cell continuation rows that belong to the target parameter.\n"
    "- DO NOT collapse min/typ/max into one 'Value' column.\n"
    "- DO NOT fabricate values. If data is absent in the context, write 'N/A'.\n"
    "- DO NOT add columns that do not exist in the <DATASHEET_TABLE> source.\n"
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
    """Build a few-shot chain-of-thought prompt for a RAG query.

    The user message contains:
      1. A hardcoded few-shot demonstration showing correct multi-row extraction.
      2. The mandatory <THINKING> CoT scaffold so the model reasons before writing.
      3. The actual retrieved context and user query.

    The few-shot example is the primary defence against "LLM laziness" — the model
    tendency to stop after the first matching row. Showing an explicit example of a
    parameter with two test conditions (25°C and 70°C) and the correct two-row output
    teaches the behaviour far more reliably than instructions alone.

    The <THINKING> block is stripped by _filter_reasoning_steps() before the
    response is returned to the user.
    """
    context = clean_latex_symbols(context)

    # ── Few-shot examples ─────────────────────────────────────────────────────
    # Example A: all rows named; Example B: blank-cell continuation (the real
    # bug — PDF datasheets only name a parameter once, further condition rows
    # have blank Parameter cells that must be inherited via Rule 0).
    few_shot_example = (
        "### EXAMPLE A: ALL CONDITION ROWS NAMED\n"
        "\n"
        "[Example Context]\n"
        "<DATASHEET_TABLE>\n"
        "| Parameter | Symbol | Conditions | min | typ | max | Unit |\n"
        "|-----------|--------|------------|-----|-----|-----|------|\n"
        "| <PARAM_A> | <SYM_A> | <COND_A> | - | - | <VAL1> | <UNIT> |\n"
        "| <PARAM_A> | <SYM_A> | <COND_B> | - | - | <VAL2> | <UNIT> |\n"
        "| <PARAM_B> | <SYM_B> | <COND_A> | - | - | <VAL3> | <UNIT> |\n"
        "| <PARAM_C> | <SYM_C> | <COND_C> | - | - | <VAL4> | <UNIT> |\n"
        "</DATASHEET_TABLE>\n"
        "\n"
        "[Example Query]\n"
        "Extract <PARAM_A>.\n"
        "\n"
        "[Example Thinking]\n"
        "<THINKING>\n"
        "1. Target Parameter: <PARAM_A>\n"
        "2. Rows found:\n"
        "   - Row 1: <PARAM_A> | <SYM_A> | <COND_A> | - | - | <VAL1> | <UNIT>  ← EXACT MATCH\n"
        "   - Row 2: <PARAM_A> | <SYM_A> | <COND_B> | - | - | <VAL2> | <UNIT>  ← EXACT MATCH\n"
        "3. Rejected rows:\n"
        "   - '<PARAM_B>' — REJECTED: different parameter name\n"
        "   - '<PARAM_C>' — REJECTED: different parameter name\n"
        "4. Source columns: Parameter | Symbol | Conditions | min | typ | max | Unit\n"
        "</THINKING>\n"
        "\n"
        "[Correct Output — one row per condition, source columns preserved]\n"
        "| Parameter | Symbol | Conditions | min | typ | max | Unit |\n"
        "|-----------|--------|------------|-----|-----|-----|------|\n"
        "| <PARAM_A> | <SYM_A> | <COND_A> | - | - | <VAL1> | <UNIT> |\n"
        "| <PARAM_A> | <SYM_A> | <COND_B> | - | - | <VAL2> | <UNIT> |\n"
        "\n"
        "[Wrong Output — DO NOT do this]\n"
        "| Parameter | Symbol | Value | Unit |\n"
        "|-----------|--------|-------|------|\n"
        "| <PARAM_A> | <SYM_A> | <VAL1> | <UNIT> |\n"
        "WRONG because: (a) the <COND_B> row is dropped, "
        "(b) min/typ/max are compressed into a fabricated 'Value' column.\n"
        "\n"
        "### EXAMPLE B: BLANK-CELL CONTINUATION (PDF TABLE CONVENTION — CRITICAL)\n"
        "Real PDF datasheets write the parameter name ONCE (first row only).\n"
        "All additional condition rows for that parameter have BLANK Parameter cells.\n"
        "Apply Rule 0: inherit the name from the nearest non-empty Parameter cell above.\n"
        "\n"
        "[Example Context — row 2 Parameter and Symbol cells are BLANK]\n"
        "<DATASHEET_TABLE>\n"
        "| Parameter | Symbol | Conditions | min | typ | max | Unit |\n"
        "|-----------|--------|------------|-----|-----|-----|------|\n"
        "| <PARAM_A> | <SYM_A> | <COND_A> | - | - | <VAL1> | <UNIT> |\n"
        "|           |         | <COND_B>  | - | - | <VAL2> | <UNIT> |\n"
        "| <PARAM_B> | <SYM_B> | <COND_C>  | - |<V3>| - | <UNIT> |\n"
        "| <PARAM_C> | <SYM_C> | <COND_D>  | - | - |<V4>| <UNIT> |\n"
        "</DATASHEET_TABLE>\n"
        "\n"
        "[Example Query] Extract <PARAM_A>.\n"
        "\n"
        "[Example Thinking]\n"
        "<THINKING>\n"
        "1. Target Parameter: <PARAM_A>\n"
        "2. Rows found:\n"
        "   - Row 1: <PARAM_A> | <SYM_A> | <COND_A> | - | - | <VAL1> | <UNIT>  "
        "<- EXACT MATCH (named)\n"
        "   - Row 2: [BLANK] | [BLANK] | <COND_B> | - | - | <VAL2> | <UNIT>  "
        "<- EXACT MATCH via Rule 0 (blank cell inherits <PARAM_A> from row above)\n"
        "3. Rejected rows:\n"
        "   - '<PARAM_B>' — REJECTED: new non-blank parameter name, row 2 was the last "
        "blank-cell continuation of <PARAM_A>\n"
        "   - '<PARAM_C>' — REJECTED: different parameter name\n"
        "4. Source columns: Parameter | Symbol | Conditions | min | typ | max | Unit\n"
        "</THINKING>\n"
        "\n"
        "[Correct Output — BOTH rows included, blank cells filled with inherited name]\n"
        "| Parameter | Symbol | Conditions | min | typ | max | Unit |\n"
        "|-----------|--------|------------|-----|-----|-----|------|\n"
        "| <PARAM_A> | <SYM_A> | <COND_A> | - | - | <VAL1> | <UNIT> |\n"
        "| <PARAM_A> | <SYM_A> | <COND_B> | - | - | <VAL2> | <UNIT> |\n"
        "\n"
        "[Wrong Output — NEVER do this]\n"
        "| Parameter | Symbol | Conditions | min | typ | max | Unit |\n"
        "|-----------|--------|------------|-----|-----|-----|------|\n"
        "| <PARAM_A> | <SYM_A> | <COND_A> | - | - | <VAL1> | <UNIT> |\n"
        "| <PARAM_B> | <SYM_B> | <COND_C> | - | <V3> | - | <UNIT> |\n"
        "WRONG because: blank-cell row 2 was skipped and <PARAM_B> (a completely "
        "DIFFERENT parameter) was substituted. This is the exact bug that produces "
        "'Gate-source leakage current' when the user asked for 'Drain-source leakage "
        "current'. NEVER substitute the next named parameter for a blank-cell row.\n"
    )

    # ── Real context + user query ─────────────────────────────────────────────
    user_content = (
        few_shot_example
        + "\n"
        + "─" * 60 + "\n"
        + "### NOW DO THE SAME FOR THE REAL DATASHEET\n\n"
        + "### DATASHEET CONTEXT\n"
        + "The following tables are from a real semiconductor datasheet. "
        + "Each table is wrapped in <DATASHEET_TABLE> tags with its exact column headers.\n\n"
        + f"{context}\n\n"
        + "### USER QUERY\n"
        + f"{query}\n\n"
        + "### YOUR PROTOCOL\n"
        + "Step 1 — Open a <THINKING> block and complete all 4 items:\n"
        + "  1. Target Parameter: write the exact parameter name from the query.\n"
        + "  2. Rows found: list (a) rows with an exactly matching Parameter cell, AND "
        + "(b) rows with a BLANK Parameter cell that inherit the target name via Rule 0. "
        + "A parameter with N test conditions will appear N times (some as blank-cell rows).\n"
        + "  3. Rejected rows: list every row you discard and WHY. "
        + "Example: 'Gate-source leakage current' — REJECTED: different non-blank parameter.\n"
        + "  4. Source columns: copy the exact column headers from the <DATASHEET_TABLE> tag.\n"
        + "Step 2 — Close </THINKING>.\n"
        + "Step 3 — Output ONLY a Markdown table containing ALL confirmed rows.\n"
        + "  - Use the EXACT source column headers (from Step 4 of your thinking).\n"
        + "  - Fill blank Parameter/Symbol cells with the inherited name/symbol.\n"
        + "  - DO NOT compress min/typ/max into a single Value column.\n"
        + "  - One test condition = one row. Never merge conditions.\n"
        + "  - CRITICAL: Do NOT stop after the first row. Output ALL matched rows "
        + "(including blank-cell continuation rows), then end the table.\n"
        + "  - No prose before or after the table.\n"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
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


def json_to_markdown(json_string: str) -> str:
    """Convert a JSON array of objects to a Markdown pipe-table.

    This is both a public utility (for use by scripts / tests) and a recovery
    mechanism: if the LLM emits JSON instead of a Markdown table, this function
    transparently converts it before the response reaches the frontend.

    The column headers are derived DYNAMICALLY from the keys of the first object,
    so this works for any parameter in any datasheet — no hardcoded columns.

    Parameters
    ----------
    json_string:
        A string containing a JSON array of flat objects, e.g.::

            [
              {"Parameter": "ID", "Conditions": "T=25C", "max": "0.23", "Unit": "A"},
              {"Parameter": "ID", "Conditions": "T=70C", "max": "0.18", "Unit": "A"}
            ]

    Returns
    -------
    str
        A GFM Markdown table if parsing succeeds, otherwise the original string.
    """
    import json as _json
    import re as _re

    # ── Step 1: extract the JSON array from the string ────────────────────────
    # The model may wrap the JSON in prose or code fences; extract the array.
    text = json_string.strip()

    # Strip ```json ... ``` or ``` ... ``` fences if present
    fence_match = _re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Find the first [ ... ] block
    bracket_match = _re.search(r"(\[\s*\{[\s\S]*\}\s*\])", text)
    if bracket_match:
        text = bracket_match.group(1)

    # ── Step 2: parse ─────────────────────────────────────────────────────────
    try:
        data = _json.loads(text)
    except (_json.JSONDecodeError, ValueError):
        # Not valid JSON — return original string unchanged
        return json_string

    if not isinstance(data, list) or not data:
        return json_string

    # Ensure all items are dicts
    rows = [item for item in data if isinstance(item, dict)]
    if not rows:
        return json_string

    # ── Step 3: derive headers from the union of all keys ────────────────────
    # Preserve insertion order from the first object, then add any extra keys
    # from subsequent objects (handles inconsistent key sets gracefully).
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            seen[key] = None
    headers = list(seen)

    # ── Step 4: build Markdown table ─────────────────────────────────────────
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep)     + " |",
    ]
    for row in rows:
        cells = [str(row.get(h, "")).replace("|", "\\|") for h in headers]
        lines.append("| " + " | ".join(cells) + " |")

    logger.debug(
        "json_to_markdown: converted %d-row JSON array → Markdown table (cols=%s)",
        len(rows), headers,
    )
    return "\n".join(lines)


def _filter_reasoning_steps(text: str) -> str:
    """Strip all CoT / thinking blocks from model output before returning to user.

    Handles four kinds of reasoning block / output format:
      1. Our own <THINKING>...</THINKING> (Chain-of-Thought protocol)
      2. Qwen's native <think>...</think> internal reasoning mode
      3. Legacy heading-based reasoning patterns ("Thinking Process:", "Step 1:", ...)
      4. JSON array output — converted to Markdown via json_to_markdown()
    """
    import re

    # ── Priority 1: strip our CoT <THINKING>...</THINKING> blocks ────────────
    # This is the block the model is explicitly instructed to produce.
    # Users should never see it — only the final Markdown table.
    text = re.sub(r"<THINKING>[\s\S]*?</THINKING>", "", text, flags=re.IGNORECASE).strip()

    # ── Priority 2: strip Qwen's native <think>...</think> blocks ────────────
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()

    # ── Priority 3: legacy "Thinking Process:" / "Draft N:" pattern ──────────
    if re.search(r"(?i)thinking process|analyze the request", text):
        match = re.search(r"(?im)(?:draft\s*\d+|final answer|answer):(.*)", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # ── Priority 4: remove numbered / bulleted reasoning section headers ──────
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
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # ── Priority 5: JSON → Markdown conversion ────────────────────────────────
    # If the model emitted a JSON array (intentionally or by mistake), convert
    # it to a Markdown pipe-table that the React frontend can render directly.
    # json_to_markdown() returns the original string unchanged on parse failure
    # so this is always safe to call.
    if cleaned.lstrip().startswith("[") or "```json" in cleaned or "```\n[" in cleaned:
        cleaned = json_to_markdown(cleaned)

    return cleaned


def generate_response(
    prompt: list[dict] | str,
    model_id: str = MODEL_NAME,
    max_new_tokens: int = 1200,
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
    max_new_tokens: int = 1200,
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
