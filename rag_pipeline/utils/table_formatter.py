"""
rag_pipeline/utils/table_formatter.py
──────────────────────────────────────────────────────────────────────────────
Table-row → readable-text converter for semiconductor datasheet tables.

Problem
-------
Docling parses PDF tables into a cell-grid structure.  When stored verbatim
those rows produce poor embeddings because the text looks like:

    "col0: Parameter | col1: Symbol | col2: 200 | col3: mA"

A vector-search model cannot match that against a natural query like
"what is the maximum drain current?".

Solution
--------
``format_table_rows()`` converts every data row into a structured natural-
language paragraph **without modifying any existing function**.

Output example
--------------
    Parameter: Continuous Drain Current
    Symbol: ID
    Minimum: —
    Typical: —
    Maximum: 200 mA
    Condition: TA = 25°C

Each converted row becomes an independent text chunk that embeds well and
retrieves cleanly.

Integration
-----------
Call ``format_table_rows()`` on the Docling table ``dict`` *before* or
*after* the existing ``chunk_table()`` — the two are fully independent.
See ``rag_pipeline/utils/table_aware_ingest.py`` for the recommended wiring.

Design constraints
------------------
* Does NOT modify any existing function or signature.
* Does NOT import from ``ingestion.datasheet_chunker`` (avoids circular deps).
* Pure Python / stdlib — no extra dependencies.
* Thread-safe (no mutable module state).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Column-header normalisation map
#
# Maps every common spelling variant found in semiconductor datasheets to a
# canonical, human-readable label.  The canonical label is used as the key
# in the formatted output line ("Minimum: 200 mA" instead of "Min.: 200 mA").
# ─────────────────────────────────────────────────────────────────────────────

_HEADER_ALIASES: Dict[str, str] = {
    # Parameter / name column
    "parameter":        "Parameter",
    "param":            "Parameter",
    "description":      "Parameter",
    "characteristic":   "Parameter",
    "characteristics":  "Parameter",
    "spec":             "Parameter",
    "specification":    "Parameter",
    "item":             "Parameter",
    "name":             "Parameter",

    # Symbol column
    "symbol":           "Symbol",
    "sym":              "Symbol",
    "notation":         "Symbol",

    # Numeric limit columns
    "min":              "Minimum",
    "min.":             "Minimum",
    "minimum":          "Minimum",
    "min value":        "Minimum",

    "typ":              "Typical",
    "typ.":             "Typical",
    "typical":          "Typical",
    "nom":              "Typical",
    "nom.":             "Typical",
    "nominal":          "Typical",

    "max":              "Maximum",
    "max.":             "Maximum",
    "maximum":          "Maximum",
    "max value":        "Maximum",

    # Unit column
    "unit":             "Unit",
    "units":            "Unit",
    "uom":              "Unit",

    # Condition / test-condition column
    "condition":        "Condition",
    "conditions":       "Condition",
    "test condition":   "Condition",
    "test conditions":  "Condition",
    "remarks":          "Condition",
    "notes":            "Condition",
    "note":             "Condition",

    # Ratings / limits
    "rating":           "Rating",
    "ratings":          "Rating",
    "value":            "Value",
    "values":           "Value",

    # Pin-related
    "pin":              "Pin",
    "pin no":           "Pin",
    "pin number":       "Pin",
    "pin name":         "Pin Name",
    "function":         "Function",
    "i/o":              "Direction",
    "type":             "Type",
}


# ─────────────────────────────────────────────────────────────────────────────
# Section header patterns — used to detect table section subtitles that appear
# as single-cell rows inside a table (e.g. "N-Channel MOSFET, Enhancement Mode")
# ─────────────────────────────────────────────────────────────────────────────

_SECTION_CAPTION_RE = re.compile(
    r"^[A-Z][A-Za-z0-9 ,/()\-]{4,80}$"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_header(raw: str) -> str:
    """Return a canonical column label for *raw* header text.

    Falls back to title-casing the raw string if no alias matches.
    """
    key = raw.strip().lower().rstrip(".")
    return _HEADER_ALIASES.get(key, raw.strip().title())


def _is_empty_value(val: str) -> bool:
    """True if *val* represents a missing / not-applicable cell."""
    stripped = val.strip()
    return stripped in {"", "-", "—", "–", "N/A", "n/a", "NA", "na", "/", ".", "--", "---"}


def _merge_value_unit(value: str, unit: str) -> str:
    """Combine a numeric value and its unit into a single string.

    Examples
    --------
    >>> _merge_value_unit("200", "mA")
    '200 mA'
    >>> _merge_value_unit("—", "")
    '—'
    """
    value = value.strip()
    unit  = unit.strip()
    if not value or _is_empty_value(value):
        return "—"
    if unit:
        return f"{value} {unit}"
    return value


def _extract_cells(table: Dict[str, Any]) -> Optional[List[List[str]]]:
    """Extract sorted rows from a Docling table dict.

    Handles both cell-grid format (``table_cells``) and pre-flattened row
    lists, returning ``None`` if the table cannot be parsed.

    Parameters
    ----------
    table:
        A single Docling ``tables[i]`` dict, as found in the parsed JSON.

    Returns
    -------
    list[list[str]] or None
        Outer list = rows (row 0 = headers), inner list = cell texts.
    """
    cells = table.get("data", {}).get("table_cells", [])

    if not cells:
        # Some Docling versions emit pre-flattened rows under "rows"
        rows = table.get("rows") or table.get("data", {}).get("rows")
        if rows and isinstance(rows, list):
            return [[str(cell).strip() for cell in row] for row in rows]
        return None

    rows_map: Dict[int, list] = {}
    for c in cells:
        r = c.get("row_index", c.get("start_row_offset_idx", 0))
        rows_map.setdefault(r, []).append(c)

    sorted_rows = [
        [
            c.get("text", "").strip()
            for c in sorted(
                rows_map[r],
                key=lambda x: x.get("col_index", x.get("start_col_offset_idx", 0)),
            )
        ]
        for r in sorted(rows_map)
    ]

    return sorted_rows if len(sorted_rows) >= 2 else None


def _is_section_subtitle(row: List[str]) -> bool:
    """True if *row* looks like a section caption row (e.g. "N-Channel MOSFET").

    These rows are single non-empty cells spanning the table width — they
    are used as sub-section labels in datasheets, not parameter data.
    """
    non_empty = [c for c in row if c.strip()]
    if len(non_empty) != 1:
        return False
    return bool(_SECTION_CAPTION_RE.match(non_empty[0]))


def _find_unit_column(headers: List[str]) -> Optional[int]:
    """Return the column index of the Unit column, or None."""
    for i, h in enumerate(headers):
        if _normalise_header(h) == "Unit":
            return i
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def format_table_rows(
    table_data: Any,
    section_name: str = "",
    part_number: str = "",
    table_number: int = 0,
) -> List[str]:
    """Convert a parsed datasheet table into a list of readable text chunks.

    Each non-empty data row is turned into a structured paragraph describing
    one parameter.  The result is suitable for direct embedding and retrieval.

    Parameters
    ----------
    table_data:
        Either:

        * A Docling ``tables[i]`` dict (has ``"data": {"table_cells": [...]}``).
        * A list of rows, where each row is a list of cell strings.
          Row 0 must contain the column headers.
        * A list of dicts, where each dict maps header → value
          (pre-parsed format).

    section_name:
        Human-readable label for the owning section, e.g.
        "electrical_characteristics".  Used in the preamble prefix of each
        chunk to improve embedding relevance.

    part_number:
        IC / component identifier, e.g. "IRF540N".  Prepended to chunk text
        so retrieval works across multi-document databases.

    table_number:
        Sequential table index within the document.  Stored in chunk prefix
        for debugging / traceability.

    Returns
    -------
    list[str]
        One formatted text string per meaningful data row.
        Empty tables, all-empty rows, and header-only tables return ``[]``.

    Examples
    --------
    >>> rows = [
    ...     ["Parameter", "Symbol", "Min", "Typ", "Max", "Unit", "Condition"],
    ...     ["Gate Threshold Voltage", "VGS(th)", "0.5", "", "1.5", "V", "ID=250µA"],
    ...     ["Drain Current", "ID", "", "", "200", "mA", "TA=25°C"],
    ... ]
    >>> chunks = format_table_rows(rows, "electrical_characteristics", "IRF540N")
    >>> print(chunks[0])
    [IRF540N | electrical characteristics | Table 0]
    Parameter: Gate Threshold Voltage
    Symbol: VGS(th)
    Minimum: 0.5 V
    Maximum: 1.5 V
    Condition: ID=250µA
    """

    # ── 1. Normalise input to list[list[str]] ─────────────────────────────────
    sorted_rows: Optional[List[List[str]]] = None

    if isinstance(table_data, dict):
        # Docling table dict
        sorted_rows = _extract_cells(table_data)

    elif isinstance(table_data, list) and table_data:
        first = table_data[0]

        if isinstance(first, list):
            # Already list-of-lists
            sorted_rows = [[str(c).strip() for c in row] for row in table_data]

        elif isinstance(first, dict):
            # List of row dicts: [{"Parameter": "ID", "Max": "200", ...}, ...]
            all_keys = list(first.keys())
            sorted_rows = [all_keys] + [
                [str(row.get(k, "")).strip() for k in all_keys]
                for row in table_data
            ]

    if not sorted_rows or len(sorted_rows) < 2:
        logger.debug("format_table_rows: table has < 2 rows — skipping")
        return []

    # ── 2. Extract and normalise headers ─────────────────────────────────────
    raw_headers = sorted_rows[0]

    # ── 2b. Detect two-row headers (e.g., "Values" → min. / typ. / max.) ────
    # Some datasheets use a merged parent header ("Values") spanning multiple
    # physical sub-columns ("min.", "typ.", "max."). Docling returns these as
    # separate rows.  We detect the sub-header row and merge the two rows into
    # a single flat header list so each sub-column gets its own canonical name.
    data_start_row = 1   # default: data rows start at sorted_rows[1]
    if len(sorted_rows) >= 3 and _is_subheader_row(sorted_rows[1]):
        raw_headers    = _merge_header_rows(sorted_rows[0], sorted_rows[1])
        data_start_row = 2   # skip both header row AND sub-header row
        logger.debug(
            "format_table_rows: two-row header detected — merged to %s", raw_headers
        )

    headers  = [_normalise_header(h) for h in raw_headers]
    num_cols = len(headers)

    # Locate the Unit column so values can be merged with their units
    unit_col = _find_unit_column(headers)

    # ── 3. Build chunk preamble prefix ───────────────────────────────────────
    section_label = section_name.replace("_", " ").strip() or "datasheet"
    table_label   = f"Table {table_number}" if table_number else "Table"
    part_label    = part_number.strip() if part_number else ""

    if part_label:
        prefix = f"[{part_label} | {section_label} | {table_label}]\n"
    else:
        prefix = f"[{section_label} | {table_label}]\n"

    # ── 4. Convert each data row into a structured paragraph ─────────────────
    chunks: List[str] = []
    current_subtitle: str = ""   # last seen section subtitle row
    current_param: str = ""      # last seen non-blank Parameter value (for blank-cell inheritance)
    current_symbol: str = ""     # last seen non-blank Symbol value

    # Find Parameter and Symbol column indices for inheritance tracking
    param_col_idx: Optional[int] = next(
        (i for i, h in enumerate(headers) if h == "Parameter"), None
    )
    symbol_col_idx: Optional[int] = next(
        (i for i, h in enumerate(headers) if h == "Symbol"), None
    )

    for row in sorted_rows[data_start_row:]:

        # Pad short rows
        while len(row) < num_cols:
            row.append("")

        # Skip all-empty rows
        if not any(c.strip() for c in row):
            continue

        # Detect and remember section subtitle rows (span single non-empty cell)
        if _is_section_subtitle(row):
            current_subtitle = next(c.strip() for c in row if c.strip())
            continue

        # ── Blank-cell inheritance: fill Parameter/Symbol from previous row ──
        # PDF datasheets only write the parameter name once per group of rows.
        # Subsequent condition rows have blank Parameter (and often Symbol) cells.
        # We propagate the last-seen name/symbol so every stored chunk is
        # self-contained and searchable without the LLM needing to infer context.
        if param_col_idx is not None:
            cell = row[param_col_idx].strip()
            if not _is_empty_value(cell):
                current_param = cell      # new named parameter — update tracker
            elif current_param:
                row[param_col_idx] = current_param  # inherit from above

        if symbol_col_idx is not None:
            sym_cell = row[symbol_col_idx].strip()
            if not _is_empty_value(sym_cell):
                current_symbol = sym_cell
            elif current_symbol and _is_empty_value(row[symbol_col_idx].strip()):
                row[symbol_col_idx] = current_symbol  # inherit from above

        # ── Build the parameter lines ─────────────────────────────────────────
        lines: List[str] = []

        # Try to find a unit value once so it can be appended to Min/Typ/Max
        unit_str = ""
        if unit_col is not None and unit_col < len(row):
            raw_unit = row[unit_col].strip()
            if raw_unit and not _is_empty_value(raw_unit):
                unit_str = raw_unit

        for i, val in enumerate(row):
            if i == unit_col:
                continue  # unit is merged into value fields

            header = headers[i] if i < len(headers) else f"Col{i}"
            val    = val.strip()

            if _is_empty_value(val):
                # For Min/Typ/Max, still emit a "—" so context is clear
                if header in ("Minimum", "Typical", "Maximum"):
                    lines.append(f"{header}: —")
                continue

            # Merge unit into numeric limit fields
            if header in ("Minimum", "Typical", "Maximum", "Value", "Rating"):
                val = _merge_value_unit(val, unit_str)

            lines.append(f"{header}: {val}")

        if not lines:
            continue

        # Remove trailing "—" lines for Min/Typ/Max to keep chunks clean
        # (keep at least one non-dash line)
        non_dash = [l for l in lines if not l.endswith(": —")]
        if non_dash:
            # Strip trailing dash-only entries
            while lines and lines[-1].endswith(": —"):
                lines.pop()

        if not lines:
            continue

        # ── Assemble final chunk text ─────────────────────────────────────────
        body = "\n".join(lines)
        if current_subtitle:
            chunk_text = f"{prefix}Section: {current_subtitle}\n{body}"
        else:
            chunk_text = f"{prefix}{body}"

        chunks.append(chunk_text)
        logger.debug(
            "format_table_rows: emitted chunk for row %d/%d",
            len(chunks), len(sorted_rows) - 1,
        )

    logger.info(
        "format_table_rows: %s table %d → %d row-chunks (%s rows processed)",
        part_label or "?", table_number, len(chunks), len(sorted_rows) - 1,
    )
    return chunks


def format_table_rows_bulk(
    tables: List[Any],
    section_names: Optional[List[str]] = None,
    part_number: str = "",
) -> List[str]:
    """Convenience wrapper — process a list of tables in one call.

    Parameters
    ----------
    tables:
        List of table dicts or row-lists, as returned by Docling or the
        ``chunk_document()`` pre-pass.
    section_names:
        Parallel list of section labels.  If shorter than *tables*, the
        last element is repeated.  If ``None``, "datasheet" is used for all.
    part_number:
        IC / component identifier.

    Returns
    -------
    list[str]
        Concatenated list of all row-chunks from every table.
    """
    all_chunks: List[str] = []
    section_names = section_names or []

    for i, tbl in enumerate(tables):
        sec = (
            section_names[i]
            if i < len(section_names)
            else (section_names[-1] if section_names else "datasheet")
        )
        all_chunks.extend(
            format_table_rows(tbl, section_name=sec, part_number=part_number, table_number=i + 1)
        )

    return all_chunks
