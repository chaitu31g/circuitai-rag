"""parameter_extractor.py
──────────────────────────────────────────────────────────────────────────────
Converts a Docling table dict (from the JSON export) into RAG Chunk objects.

KEY FIXES (v3):
  1. Bulletproof cell scrubbing: replaces whitespace-only strings, None, and
     empty strings with pd.NA before ffill, catching all hidden Docling
     artifacts (not just plain "").
  2. ffill only the first two columns (Parameter name + Symbol) so merged-cell
     parameters (e.g. "Continuous drain current" at 25°C AND 70°C) are never
     silently dropped.
  3. Table text is wrapped in <DATASHEET_TABLE>...</DATASHEET_TABLE> XML tags
     so the LLM prompt can reference the exact source format and is forbidden
     from inventing new column names.
"""

from __future__ import annotations

import logging
import re
from typing import List

from ingestion.datasheet_chunker import Chunk

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scrub_and_ffill(rows: list[list[str]], headers: list[str] = None) -> list[list[str]]:
    """Bulletproof merged-cell repair for Docling table rows.

    Docling inserts several kinds of "empty" values into cells that belong to
    a vertically merged parent:
      • Plain empty string ""
      • Whitespace-only strings "  ", "\\n", "\\t"
      • Python None
      • Unicode whitespace / zero-width spaces

    All of these must be treated as pd.NA before calling ffill, otherwise the
    forward-fill has no effect and the second test-condition row (e.g. 70°C)
    is dropped by the downstream "if not any(cell)" guard.

    Columns filled forward:
      • Column 0 (Parameter name) — always merged vertically for multi-condition rows
      • Column 1 (Symbol) — same
      • Last column (Unit) — Docling only sets it in the first condition row;
        continuation rows leave it blank, causing unit to disappear from chunks.
      • Conditions column — (if detected in headers) forward-fills test conditions 
        for parameters spanning multiple rows.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available; skipping merged-cell ffill.")
        return rows

    if not rows:
        return rows

    df = pd.DataFrame(rows)
    n_cols = df.shape[1]

    # Columns to forward-fill: first two (param, symbol) + last (unit)
    fill_cols = [0, 1, n_cols - 1]

    # Also forward-fill the 'Conditions' column so sub-rows inherit test parameters
    if headers:
        for i, h in enumerate(headers):
            if "condition" in h.lower() or "test" in h.lower():
                if i < n_cols:
                    fill_cols.append(i)

    fill_cols = list(dict.fromkeys(fill_cols))  # dedup in case ncols<=2

    # ── Step 1: Scrub ALL forms of empty from fill columns ────────────────────
    for col in fill_cols:
        df.iloc[:, col] = df.iloc[:, col].replace(r"^\s*$", pd.NA, regex=True)
        df.iloc[:, col] = df.iloc[:, col].replace([None, ""], pd.NA)

    # ── Step 2: Forward-fill blanks downward ──────────────────────────────────
    for col in fill_cols:
        df.iloc[:, col] = df.iloc[:, col].ffill()

    # ── Step 3: Convert any remaining NA back to "" for downstream safety ─────
    df = df.fillna("")

    return df.values.tolist()


def _rows_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """Convert headers + data rows to a GFM pipe-table Markdown string."""
    # Sanitize headers: replace empty header names with generic col labels
    clean_headers = [h if h else f"Col{i}" for i, h in enumerate(headers)]
    sep = ["-" * max(3, len(h)) for h in clean_headers]

    lines = [
        "| " + " | ".join(clean_headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        # Pad short rows
        padded = list(row) + [""] * (len(clean_headers) - len(row))
        lines.append("| " + " | ".join(str(c) for c in padded[:len(clean_headers)]) + " |")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — called from datasheet_chunker.py
# ─────────────────────────────────────────────────────────────────────────────

def extract_parameter_rows(
    table: dict,
    section_name: str,
    part_number: str,
    table_number: int,
    table_title: str = "",
) -> List[Chunk]:
    """Convert a Docling table dict into a list of RAG Chunk objects.

    Pipeline:
      1. Reconstruct row-sorted cell matrix from dict ``table_cells`` entries.
      2. Detect the header row (first row containing column-keyword words).
      3. Run ``_scrub_and_ffill`` on all data rows to repair merged cells.
      4. Emit ONE chunk per data row (parameter_row format) for fine-grained
         retrieval, PLUS one whole-table chunk (table_markdown format) so the
         LLM always receives the full column structure and can avoid inventing
         column names.
    """
    cells = table.get("data", {}).get("table_cells", [])
    if not cells:
        return []

    # ── Reconstruct row matrix (position-aligned sparse grid) ────────────────
    # IMPORTANT: We must preserve each cell's exact column index from Docling.
    # A naive dense extraction (just sorting by col_index and taking text) drops
    # gap columns — e.g. "Values" at col3 that spans to col5, with "Unit" at
    # col6, gets collapsed to positional indices 3 and 4.  After sub-header
    # merging ("min.", "typ.", "max." at indices 3,4,5) Unit would be clobbered
    # by "typ.", causing every typ value to appear in the Unit column.
    #
    # Fix: compute the full column count (max_col + 1), then build each row as
    # a ``max_col``-wide list of empty strings and fill in cells at their exact
    # col_index positions.
    rows_map: dict[int, list] = {}
    for c in cells:
        r = c.get("row_index", c.get("start_row_offset_idx", 0))
        rows_map.setdefault(r, []).append(c)

    if not rows_map:
        return []

    # Use end_col_offset_idx (exclusive end) when available so spanning cells
    # (like "Values" at start=3 spanning to end=6) correctly widen the grid.
    # Without this, "Unit" at start=4 in the densely-numbered header row ends
    # up at grid position 4, colliding with "typ." at position 4 in the sub-header.
    def _cell_end_col(c: dict) -> int:
        end = c.get("end_col_offset_idx")
        if end is not None:
            return end - 1   # exclusive → inclusive
        # Fallback: use start position (1-column wide cell)
        return c.get("col_index", c.get("start_col_offset_idx", 0))

    max_col = max(
        _cell_end_col(c)
        for row_cells in rows_map.values()
        for c in row_cells
    )
    grid_width = max_col + 1

    sorted_rows: list[list[str]] = []
    for r in sorted(rows_map):
        row: list[str] = [""] * grid_width
        for c in sorted(
            rows_map[r],
            key=lambda x: x.get("col_index", x.get("start_col_offset_idx", 0)),
        ):
            col = c.get("col_index", c.get("start_col_offset_idx", 0))
            if col < grid_width:
                row[col] = c.get("text", "").strip()
        sorted_rows.append(row)

    if not sorted_rows:
        return []

    # ── Detect header row ─────────────────────────────────────────────────────
    HEADER_KEYWORDS = {"parameter", "symbol", "typ", "max", "min",
                       "condition", "unit", "rating", "limit", "value", "test"}
    header_idx = 0
    for idx, row in enumerate(sorted_rows[:3]):
        row_text = " ".join((c or "").lower() for c in row)
        if any(kw in row_text for kw in HEADER_KEYWORDS):
            header_idx = idx
            break

    raw_headers = sorted_rows[header_idx]

    # ── Detect two-row header (e.g., "Values" parent → min./typ./max. sub) ───
    # Many datasheets use a merged parent header ("Values" or "Ratings") that
    # spans multiple sub-columns. Docling returns these as separate rows.
    # Detect and merge so each sub-column gets its own name.
    _SUBHDR = frozenset({
        "min", "min.", "minimum",
        "typ", "typ.", "typical", "nom", "nom.", "nominal",
        "max", "max.", "maximum",
        "value", "values", "val",
        "rating", "ratings", "limit", "limits",
    })

    def _is_subheader_row(row: list) -> bool:
        non_empty = [c.strip() for c in row if c.strip()]
        if not non_empty:
            return False
        matched = sum(1 for c in non_empty if c.lower().rstrip(".") in _SUBHDR)
        if matched < 2:
            return False
        # Reject if any cell is longer than 20 chars (a real parameter value)
        if any(len(c) > 20 for c in non_empty):
            return False
        return True

    data_row_start = header_idx + 1
    next_row_idx = header_idx + 1
    if next_row_idx < len(sorted_rows) and _is_subheader_row(sorted_rows[next_row_idx]):
        sub = sorted_rows[next_row_idx]

        # ── Span-token replacement merge ──────────────────────────────────────
        # The parent header may have a single merged cell like "Values" that
        # spans the min/typ/max sub-columns.  If we naively merge by list
        # position, "typ." (at sub[4]) can clobber "Unit" (at parent[4]) when
        # Docling numbers the header row densely (without gap slots for spans).
        #
        # Robust fix: locate the span token ("Values", "Ratings" …) in the
        # parent, replace it with the extracted sub-header tokens, and keep
        # every other real parent header in its original relative order.
        _SPAN_TOKENS = frozenset({
            "values", "value", "rating", "ratings", "limit", "limits",
        })
        # Only keep cells in sub that are recognised sub-header labels
        sub_tokens = [
            s.strip() for s in sub
            if s.strip() and s.strip().lower().rstrip(".") in _SUBHDR
        ]

        span_idx = next(
            (i for i, p in enumerate(raw_headers)
             if p.strip().lower() in _SPAN_TOKENS),
            None,
        )

        if span_idx is not None and sub_tokens:
            # before-span + sub-tokens + after-span (drop empty gap slots)
            before = list(raw_headers[:span_idx])
            after  = [p.strip() for p in raw_headers[span_idx + 1:] if p.strip()]
            raw_headers = before + sub_tokens + after
        else:
            # Fallback: positional merge (no span token found)
            length = max(len(raw_headers), len(sub))
            merged: list[str] = []
            for i in range(length):
                p = raw_headers[i].strip() if i < len(raw_headers) else ""
                s = sub[i].strip()         if i < len(sub)         else ""
                if s and s.lower().rstrip(".") in _SUBHDR:
                    merged.append(s)
                elif s and not p:
                    merged.append(s)
                else:
                    merged.append(p)
            raw_headers = merged

        data_row_start = next_row_idx + 1
        logger.debug(
            "extract_parameter_rows: two-row header merged → %s", raw_headers
        )

    # Handle the rare case where header cells contain merged header+data
    headers: list[str] = []
    first_data_from_header: list[str] = []
    has_merged_header_data = any(
        "\n" in cell and len(cell.split("\n")) > 1 for cell in raw_headers
    )
    if has_merged_header_data:
        for cell in raw_headers:
            lines = [ln.strip() for ln in cell.split("\n") if ln.strip()]
            headers.append(lines[0] if lines else "")
            first_data_from_header.append(" ".join(lines[1:]) if len(lines) > 1 else "")
    else:
        headers = raw_headers

    rows_to_process: list[list[str]] = []
    if first_data_from_header and any(first_data_from_header):
        rows_to_process.append(first_data_from_header)
    rows_to_process.extend(sorted_rows[data_row_start:])

    if not rows_to_process:
        return []

    # ── CRITICAL: Scrub + forward-fill merged cells ───────────────────────────
    logger.debug(
        "[ffill] table=%d  before=%d rows  headers=%s",
        table_number, len(rows_to_process), headers[:4],
    )
    rows_to_process = _scrub_and_ffill(rows_to_process, headers)
    logger.debug("[ffill] table=%d  after =%d rows", table_number, len(rows_to_process))

    # ── Build Markdown table (wrapped in XML tags for LLM prompt coherence) ───
    # This is the "whole table" representation emitted as a single chunk.
    # The XML tags tell the LLM: "these column names come from the source —
    # do NOT invent new column names like Min/Typ/Max."
    markdown_table = _rows_to_markdown(headers, rows_to_process)
    table_xml = f"<DATASHEET_TABLE>\n{markdown_table}\n</DATASHEET_TABLE>"

    section_label = section_name.replace("_", " ").title()
    table_label = table_title or f"Table {table_number}"
    page = (table.get("prov") or [{}])[0].get("page_no")
    num_cols = len(headers)

    chunks: List[Chunk] = []

    # ── Chunk A: Whole-table Markdown chunk ───────────────────────────────────
    # Kept intact — MarkdownNodeParser or the embedding model will NOT split it
    # because it is a single document unit. The LLM prompt is built from this
    # so it always sees all column headers.
    table_preamble = (
        f"Section: {section_label}\n"
        f"Table: {table_label}\n"
        f"Component: {part_number}\n"
        f"Columns present: {', '.join(h for h in headers if h)}\n\n"
    )
    chunks.append(Chunk(
        text=table_preamble + table_xml,
        chunk_type="table_markdown",
        metadata={
            "part_number":  part_number,
            "section_name": section_name,
            "table_number": table_number,
            "table_title":  table_label,
            "page":         page,
            "num_rows":     len(rows_to_process),
            "num_cols":     num_cols,
            "chunk_type":   "table_markdown",
        },
    ))

    # ── Chunk B: Per-row parameter chunks ─────────────────────────────────────
    # One chunk per data row for fine-grained vector search.
    # Each chunk includes the full column list so the LLM never loses context.
    unit_col_idx = next(
        (i for i, h in enumerate(headers) if "unit" in str(h).lower()), -1
    )

    for row in rows_to_process:
        while len(row) < num_cols:
            row.append("")

        if not any(cell for cell in row):
            continue

        unit_str = (
            row[unit_col_idx].strip()
            if unit_col_idx != -1 and unit_col_idx < len(row)
            else ""
        )

        row_data: dict[str, str] = {}
        param_value = ""
        symbol_value = ""
        is_meaningful = False

        for i, val in enumerate(row):
            if i == unit_col_idx:
                continue
            if not val or val in ("-", "n/a", "N/A"):
                continue

            hdr_orig = headers[i] if i < len(headers) else f"Column{i}"
            hdr_lower = hdr_orig.lower()

            # Normalise common column names for metadata
            if "parameter" in hdr_lower:
                hdr_display = "Parameter"
                param_value = val
            elif "symbol" in hdr_lower:
                hdr_display = "Symbol"
                symbol_value = val
            elif "condition" in hdr_lower:
                hdr_display = "Conditions"
            elif "min" in hdr_lower:
                hdr_display = "Minimum"
            elif "max" in hdr_lower:
                hdr_display = "Maximum"
            elif "typ" in hdr_lower:
                hdr_display = "Typical"
            elif "value" in hdr_lower:
                hdr_display = "Value"
            else:
                hdr_display = hdr_orig  # preserve original column name exactly

            row_data[hdr_display] = val
            if val.lower() not in {"parameter", "symbol", "condition",
                                    "minimum", "maximum", "typical", "value"}:
                is_meaningful = True

        if not is_meaningful or not row_data:
            continue

        # Build chunk text — include column names so the LLM knows the schema
        chunk_lines = [
            f"Section: {section_label}",
            f"Table: {table_label}",
            f"Component: {part_number}",
            f"Table columns: {', '.join(h for h in headers if h)}",
        ]

        # Parameter + Symbol first for high embedding weight
        if "Parameter" in row_data:
            chunk_lines.append(f"Parameter: {row_data.pop('Parameter')}")
        if "Symbol" in row_data:
            chunk_lines.append(f"Symbol: {row_data.pop('Symbol')}")

        for key in ("Minimum", "Typical", "Maximum", "Value"):
            if key in row_data:
                chunk_lines.append(f"{key}: {row_data.pop(key)}")

        if unit_str and unit_str.lower() != "unit":
            chunk_lines.append(f"Unit: {unit_str}")

        if "Conditions" in row_data:
            chunk_lines.append(f"Conditions: {row_data.pop('Conditions')}")

        for key, val in row_data.items():
            chunk_lines.append(f"{key}: {val}")

        chunks.append(Chunk(
            text="\n".join(chunk_lines),
            chunk_type="parameter_row",
            metadata={
                "type":        "table_row",
                "section":     section_label,
                "table_name":  table_label,
                "parameter":   param_value,
                "symbol":      symbol_value,
                "page":        page,
                "chunk_type":  "parameter_row",
                "part_number": part_number,
                "table_index": table_number,
            },
        ))

    return chunks
