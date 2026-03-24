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

def _scrub_and_ffill(rows: list[list[str]]) -> list[list[str]]:
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

    Only the first two columns (Parameter name + Symbol) are filled — all
    other columns retain their original blanks so value cells are not smeared.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available; skipping merged-cell ffill.")
        return rows

    if not rows:
        return rows

    df = pd.DataFrame(rows)

    # ── Step 1: Scrub ALL forms of empty from first two cols ──────────────────
    # regex=True catches whitespace-only strings ("  ", "\n", "\t", etc.)
    df.iloc[:, :2] = df.iloc[:, :2].replace(r"^\s*$", pd.NA, regex=True)
    # Also catch Python None and literal empty string missed by regex
    df.iloc[:, :2] = df.iloc[:, :2].replace([None, ""], pd.NA)

    # ── Step 2: Forward-fill blanks downward ──────────────────────────────────
    df.iloc[:, :2] = df.iloc[:, :2].ffill(axis=0)

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

    # ── Reconstruct row matrix ────────────────────────────────────────────────
    rows_map: dict[int, list] = {}
    for c in cells:
        r = c.get("row_index", c.get("start_row_offset_idx", 0))
        rows_map.setdefault(r, []).append(c)

    sorted_rows = [
        [c.get("text", "").strip()
         for c in sorted(rows_map[r],
                         key=lambda x: x.get("col_index", x.get("start_col_offset_idx", 0)))]
        for r in sorted(rows_map)
    ]

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
    rows_to_process.extend(sorted_rows[header_idx + 1:])

    if not rows_to_process:
        return []

    # ── CRITICAL: Scrub + forward-fill merged cells ───────────────────────────
    logger.debug(
        "[ffill] table=%d  before=%d rows  headers=%s",
        table_number, len(rows_to_process), headers[:4],
    )
    rows_to_process = _scrub_and_ffill(rows_to_process)
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
