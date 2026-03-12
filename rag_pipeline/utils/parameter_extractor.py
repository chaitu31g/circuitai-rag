import logging
import re
from typing import List
from ingestion.datasheet_chunker import Chunk

logger = logging.getLogger(__name__)

def extract_parameter_rows(table: dict, section_name: str, part_number: str, table_number: int) -> List[Chunk]:
    """
    Extracts structured parameter records from a table.
    Addresses issues where the first row is merged with headers or skipped.
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

    # 1. Detect the actual header row (it's not always row 0, sometimes row 0 is a table title)
    header_idx = 0
    for idx, row in enumerate(sorted_rows):
        row_text = " ".join((c or "").lower() for c in row)
        if "parameter" in row_text or "symbol" in row_text or "typ" in row_text or "max" in row_text or "condition" in row_text:
            header_idx = idx
            break

    raw_headers = sorted_rows[header_idx]
    headers = []
    first_data_row = []
    
    # Check if the header row has merged data (e.g. "Parameter\nInput capacitance")
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

    num_cols = len(headers)
    unit_col_idx = -1
    for i, h in enumerate(headers):
        if "unit" in str(h).lower():
            unit_col_idx = i
            break

    # Reconstruct the rows to process, discarding anything before the header row
    rows_to_process = []
    if first_data_row and any(first_data_row):
        rows_to_process.append(first_data_row)
        
    for i, row in enumerate(sorted_rows[header_idx:]):
        if i == 0:
            if has_merged_data:
                continue # We already generated first_data_row
            
            # If not merged, ensure it's not purely a header
            is_just_headers = True
            for val in row:
                val_lower = str(val).lower()
                if "ciss" in val_lower or "coss" in val_lower or "crss" in val_lower or re.search(r'\d+', str(val)):
                    is_just_headers = False
                    break
            if is_just_headers:
                continue 
                
        rows_to_process.append(row)

    page = (table.get("prov") or [{}])[0].get("page_no")
    chunks: List[Chunk] = []

    # Process all identified data rows
    for row in rows_to_process:
        while len(row) < num_cols:
            row.append("")

        if not any(cell for cell in row):
            continue

        unit_str = ""
        if unit_col_idx != -1 and unit_col_idx < len(row):
            unit_str = row[unit_col_idx].strip()

        parts = []
        raw_vals_for_log = []
        is_meaningful_param = False

        for i, val in enumerate(row):
            if i == unit_col_idx:
                raw_vals_for_log.append(val)
                continue
                
            if not val or val == "-" or val.lower() == "n/a":
                raw_vals_for_log.append("-")
                continue

            hdr_orig = headers[i] if i < len(headers) else f"Column {i}"
            hdr_lower = hdr_orig.lower()

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
            elif "typ" in hdr_lower:
                hdr_display = "Typical"
            elif "value" in hdr_lower:
                hdr_display = "Value"
            else:
                # If header is completely missing and it's a numeric column at the end
                if hdr_orig == "" or hdr_orig.startswith("Column"):
                    if i == num_cols - 4:
                        hdr_display = "Minimum"
                    elif i == num_cols - 3:
                        hdr_display = "Typical"
                    elif i == num_cols - 2:
                        hdr_display = "Maximum"
                    else:
                        hdr_display = f"Column {i}"
                else:
                    hdr_display = hdr_orig

            if unit_str and hdr_display in ["Value", "Minimum", "Maximum", "Typical", "Limit"]:
                val_str = f"{val} {unit_str}".strip()
            else:
                val_str = val

            parts.append(f"{hdr_display}: {val_str}")
            raw_vals_for_log.append(val)
            
            # Identify if it's an actual parameter row rather than a sub-header
            if val_str and val_str.lower() not in ["parameter", "symbol", "condition", "minimum", "maximum", "typical", "value"]:
                is_meaningful_param = True

        if is_meaningful_param and parts:
            # Example logging: Extracted table row: Input capacitance | Ciss | 32 | 41 | pF
            log_line = " | ".join([v for v in raw_vals_for_log if v and v != "-"])
            if unit_str and not log_line.endswith(unit_str):
                log_line += f" | {unit_str}"
            logger.info(f"Extracted table row: {log_line}")
            
            parts.append(f"Section: {section_name}")
            text = "\n".join(parts)
            
            metadata = {
                "component_name": part_number,
                "part_number": part_number,
                "section": section_name,
                "section_name": section_name,
                "table_index": table_number,
                "table_number": table_number,
                "page_number": page,
                "page": page,
                "chunk_type": "parameter_row"
            }
            
            chunks.append(Chunk(
                text=text,
                chunk_type="parameter_row",
                metadata=metadata
            ))

    return chunks
