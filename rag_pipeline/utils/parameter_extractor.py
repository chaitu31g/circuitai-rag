import logging
import re
from typing import List
from ingestion.datasheet_chunker import Chunk

logger = logging.getLogger(__name__)


def _ffill_merged_cells(rows: list[list[str]]) -> list[list[str]]:
    """Forward-fill the first two columns (Parameter name + Symbol) to fix
    merged-cell data loss.

    Datasheets use merged cells for parameters with multiple test conditions,
    e.g. "Continuous drain current" spans rows for 25°C and 70°C, but the
    second row's parameter cell is blank in the parsed PDF.

    Equivalent to:  df.iloc[:, :2].ffill(axis=0)
    """
    try:
        import pandas as pd
        import numpy as np

        if not rows:
            return rows

        df = pd.DataFrame(rows)
        # Replace empty strings with NaN so ffill works correctly
        df.iloc[:, :2] = df.iloc[:, :2].replace("", np.nan)
        df.iloc[:, :2] = df.iloc[:, :2].ffill(axis=0)
        # Convert NaN back to empty strings for downstream logic
        df = df.fillna("")
        return df.values.tolist()
    except ImportError:
        # pandas/numpy not available — skip ffill silently
        logger.warning("pandas not available; skipping merged-cell forward-fill.")
        return rows


def extract_parameter_rows(table: dict, section_name: str, part_number: str, table_number: int, table_title: str = "") -> List[Chunk]:
    """
    Extracts structured parameter records from a table.

    Key fix: applies ffill to the first two columns (Parameter, Symbol) before
    processing rows, so condition rows belonging to merged-cell parameters are
    not silently dropped due to a blank parameter name.

    Includes table_title and section context in every row-level chunk.
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

    sorted_rows = [
        [c.get("text", "").strip()
         for c in sorted(rows_map[r], key=lambda x: x.get("col_index", x.get("start_col_offset_idx", 0)))]
        for r in sorted(rows_map)
    ]

    if not sorted_rows:
        return []

    # 1. Detect the actual header row
    header_idx = -1
    for idx, row in enumerate(sorted_rows[:3]): # Check first 3 rows
        row_text = " ".join((c or "").lower() for c in row)
        if any(x in row_text for x in ["parameter", "symbol", "typ", "max", "min", "condition", "unit", "rating", "limit"]):
            header_idx = idx
            break

    if header_idx == -1:
        # Fallback: if first column of many rows is a known parameter name, assume no header row
        header_idx = 0 
        # But we don't want to skip it if it's data. 
        # Actually, let's check if Row 0 looks like headers or data.
        row0_text = " ".join((c or "").lower() for c in sorted_rows[0])
        # If row 0 contains values/numbers, it's likely data, and headers are missing.
        if re.search(r'\d+', row0_text):
            # Headers are missing, use generic headers
            headers = ["Parameter", "Value", "Unit", "Typical", "Maximum"] # Greedy guess
            rows_to_process = sorted_rows
        else:
            headers = sorted_rows[0]
            rows_to_process = sorted_rows[1:]
    else:
        raw_headers = sorted_rows[header_idx]
        headers = []
        first_data_row = []
        
        # Check if the header row has merged data
        has_merged_data = any("\n" in cell and len(cell.split("\n")) > 1 for cell in raw_headers)
        
        if has_merged_data:
            for cell in raw_headers:
                lines = [line.strip() for line in cell.split('\n') if line.strip()]
                if len(lines) >= 2:
                    headers.append(lines[0])
                    first_data_row.append(" ".join(lines[1:]))
                elif len(lines) == 1:
                    headers.append(lines[0])
                    first_data_row.append("")
                else:
                    headers.append("")
                    first_data_row.append("")
        else:
            headers = raw_headers

        rows_to_process = []
        if first_data_row and any(first_data_row):
            rows_to_process.append(first_data_row)
            
        for i, row in enumerate(sorted_rows[header_idx + 1:]):
            rows_to_process.append(row)

    num_cols = len(headers)
    unit_col_idx = -1
    for i, h in enumerate(headers):
        if "unit" in str(h).lower():
            unit_col_idx = i
            break

    page = (table.get("prov") or [{}])[0].get("page_no")
    chunks: List[Chunk] = []

    # ── KEY FIX: Forward-fill merged parameter-name cells ─────────────────────
    # Many datasheet tables use vertically merged cells for parameters that have
    # multiple test conditions (e.g. 25°C and 70°C rows for the same parameter).
    # Docling parses the second+ rows with a blank first column, so without ffill
    # those rows would be silently dropped by the `not any(cell)` guard below.
    # We only ffill the first two columns (Parameter name + Symbol).
    print(f"[DEBUG ffill] rows_to_process before ffill: {len(rows_to_process)} rows")
    rows_to_process = _ffill_merged_cells(rows_to_process)
    print(f"[DEBUG ffill] rows_to_process after  ffill: {len(rows_to_process)} rows")

    # Process all identified data rows
    for row in rows_to_process:
        while len(row) < num_cols:
            row.append("")

        if not any(cell for cell in row):
            continue

        unit_str = ""
        if unit_col_idx != -1 and unit_col_idx < len(row):
            unit_str = row[unit_col_idx].strip()

        row_data = {}
        is_meaningful_param = False
        param_value = ""
        symbol_value = ""

        for i, val in enumerate(row):
            if i == unit_col_idx:
                continue
                
            if not val or val == "-" or val.lower() == "n/a":
                continue

            hdr_orig = headers[i] if i < len(headers) else f"Column {i}"
            hdr_lower = hdr_orig.lower()

            hdr_display = hdr_orig # Default
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
            
            row_data[hdr_display] = val
            
            if val and val.lower() not in ["parameter", "symbol", "condition", "minimum", "maximum", "typical", "value"]:
                is_meaningful_param = True

        if is_meaningful_param and row_data:
            # Build chunk text in the specific format requested
            chunk_lines = []
            chunk_lines.append(f"Section: {section_name.replace('_', ' ').title()}")
            chunk_lines.append(f"Table: {table_title or f'Table {table_number}'}")
            
            # Ensure Parameter and Symbol are at the top if present
            if "Parameter" in row_data:
                chunk_lines.append(f"Parameter: {row_data.pop('Parameter')}")
            if "Symbol" in row_data:
                chunk_lines.append(f"Symbol: {row_data.pop('Symbol')}")
            
            # Add other numeric/limit fields
            for key in ["Minimum", "Typical", "Maximum", "Value"]:
                if key in row_data:
                    chunk_lines.append(f"{key}: {row_data.pop(key)}")
            
            # Add Unit if found
            if unit_str and unit_str.lower() != "unit":
                chunk_lines.append(f"Unit: {unit_str}")
                
            # Add Conditions
            if "Conditions" in row_data:
                chunk_lines.append(f"Conditions: {row_data.pop('Conditions')}")
            
            # Add any remaining fields
            for key, val in row_data.items():
                chunk_lines.append(f"{key}: {val}")
                
            text = "\n".join(chunk_lines)
            
            metadata = {
                "type": "table_row",
                "section": section_name.replace('_', ' ').title(),
                "table_name": table_title or f"Table {table_number}",
                "parameter": param_value,
                "symbol": symbol_value,
                "page": page,
                "chunk_type": "parameter_row",
                "part_number": part_number,
                "table_index": table_number
            }
            
            chunks.append(Chunk(
                text=text,
                chunk_type="parameter_row",
                metadata=metadata
            ))

    return chunks
