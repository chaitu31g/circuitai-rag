from typing import List
import re
from ingestion.datasheet_chunker import Chunk

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

    headers = [h.strip() for h in sorted_rows[0]]
    num_cols = len(headers)
    
    # Identify unit column
    unit_col_idx = -1
    for i, h in enumerate(headers):
        if "unit" in h.lower():
            unit_col_idx = i
            break

    page = (table.get("prov") or [{}])[0].get("page_no")
    chunks: List[Chunk] = []

    for row in sorted_rows[1:]:
        # Pad row to match headers
        while len(row) < num_cols:
            row.append("")

        # Skip empty rows
        if not any(cell for cell in row):
            continue

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
            elif "typ" in hdr_lower:
                hdr_display = "Typical"
            elif "value" in hdr_lower:
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
            # Add prefix context to improve embedding? We won't strictly add it unless it helps, 
            # but usually just the table text is sufficient if chunk metadata is used, 
            # however it's a good practice to include part number in the text.
            # Wait! The example from user specifically requested exactly:
            # Parameter: Continuous drain current
            # Symbol: ID
            # Condition: TA = 25°C
            # Value: 0.23 A
            # So I will not add extra strings that pollute the chunk text.
            
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
