from typing import List
import re
from ingestion.datasheet_chunker import Chunk

# Known sub-header tokens for multi-row header detection
_SUBHDR = frozenset({
    "min", "min.", "minimum",
    "typ", "typ.", "typical", "nom", "nom.", "nominal",
    "max", "max.", "maximum",
    "value", "values", "val",
    "rating", "ratings", "limit", "limits",
})


def _is_subheader_row(row: list) -> bool:
    """True if row looks like a sub-header (e.g., min. / typ. / max.)."""
    non_empty = [c.strip() for c in row if c.strip()]
    if not non_empty:
        return False
    matched = sum(1 for c in non_empty if c.lower().rstrip(".") in _SUBHDR)
    if matched < 2:
        return False
    if any(len(c) > 20 for c in non_empty):
        return False
    return True


def _merge_header_rows(parent: list, sub: list) -> list:
    """Merge parent + sub-header rows into one flat header list."""
    length = max(len(parent), len(sub))
    merged = []
    for i in range(length):
        p = parent[i].strip() if i < len(parent) else ""
        s = sub[i].strip()    if i < len(sub)    else ""
        if s and s.lower().rstrip(".") in _SUBHDR:
            merged.append(s)
        elif s and not p:
            merged.append(s)
        else:
            merged.append(p)
    return merged


def format_table_rows(table: dict, section_name: str, part_number: str, table_number: int) -> List[Chunk]:
    """
    Transforms a parsed table into a list of row-level parameter chunks.
    Extracts headers, iterates through rows, skipping empty ones, and maps
    them into a structured Parameter text representation.
    """
    cells = table.get("data", {}).get("table_cells", [])
    if not cells:
        return []

    rows_map: dict[int, list] = {}
    for c in cells:
        r = c.get("row_index", c.get("start_row_offset_idx", 0))
        if r not in rows_map:
            rows_map[r] = []
        rows_map[r].append(c)

    # Sort rows and columns
    sorted_rows = [
        [c.get("text", "").strip()
         for c in sorted(rows_map[r], key=lambda x: x.get("col_index", x.get("start_col_offset_idx", 0)))]
        for r in sorted(rows_map)
    ]

    if len(sorted_rows) < 2:
        return []

    # ── Detect two-row headers (e.g., "Values" → min. / typ. / max.) ─────────
    data_start = 1
    raw_headers = sorted_rows[0]
    if len(sorted_rows) >= 3 and _is_subheader_row(sorted_rows[1]):
        raw_headers = _merge_header_rows(sorted_rows[0], sorted_rows[1])
        data_start  = 2   # skip the sub-header row in data processing

    headers = [h.strip() for h in raw_headers]
    num_cols = len(headers)

    # Identify unit column
    unit_col_idx = -1
    for i, h in enumerate(headers):
        if "unit" in h.lower():
            unit_col_idx = i
            break

    page = (table.get("prov") or [{}])[0].get("page_no")
    chunks: List[Chunk] = []

    # ── Blank-cell inheritance tracking ──────────────────────────────────────
    param_col_idx = next((i for i, h in enumerate(headers) if "parameter" in h.lower()), None)
    symbol_col_idx = next((i for i, h in enumerate(headers) if "symbol" in h.lower()), None)
    current_param = ""
    current_symbol = ""

    for row in sorted_rows[data_start:]:
        # Pad row to match headers
        while len(row) < num_cols:
            row.append("")

        # Skip empty rows
        if not any(cell for cell in row):
            continue

        # ── Blank-cell inheritance ────────────────────────────────────────────
        if param_col_idx is not None:
            cell = row[param_col_idx].strip()
            if cell and cell not in {"-", "—", "N/A"}:
                current_param = cell
            elif current_param:
                row[param_col_idx] = current_param

        if symbol_col_idx is not None:
            sym = row[symbol_col_idx].strip()
            if sym and sym not in {"-", "—", "N/A"}:
                current_symbol = sym
            elif current_symbol:
                row[symbol_col_idx] = current_symbol

        unit_str = ""
        if unit_col_idx != -1 and unit_col_idx < len(row):
            unit_str = row[unit_col_idx].strip()

        parts = []
        for i, val in enumerate(row):
            if i == unit_col_idx:
                continue
                
            if not val or val == "-" or val.lower() == "n/a":
                continue

            hdr_orig = headers[i] if i < len(headers) else f"Column {i}"
            hdr_lower = hdr_orig.lower()

            # Clean up the header for display
            if "parameter" in hdr_lower:
                hdr_display = "Parameter"
            elif "symbol" in hdr_lower:
                hdr_display = "Symbol"
            elif "condition" in hdr_lower:
                hdr_display = "Condition"
            elif "min" in hdr_lower:
                hdr_display = "Minimum"
            elif "max" in hdr_lower:
                hdr_display = "Maximum"
            elif "typ" in hdr_lower or "nom" in hdr_lower:
                hdr_display = "Typical"
            elif "value" in hdr_lower or "val" in hdr_lower:
                hdr_display = "Value"
            else:
                hdr_display = hdr_orig or f"Column {i}"

            # If it's a numeric/limit field, append unit
            if unit_str and hdr_display in ["Value", "Minimum", "Maximum", "Typical", "Limit"]:
                val_str = f"{val} {unit_str}".strip()
            else:
                val_str = val

            parts.append(f"{hdr_display}: {val_str}")

        if parts:
            text = "\n".join(parts)
            metadata = {
                "type": "table_row",
                "component_name": part_number,
                "part_number": part_number,
                "section": section_name,
                "section_name": section_name,
                "table_index": table_number,
                "table_number": table_number,
                "page_number": page,
                "page": page,
                "chunk_type": "table_row"
            }
            
            chunks.append(Chunk(
                text=text,
                chunk_type="table_row",
                metadata=metadata
            ))

    return chunks
