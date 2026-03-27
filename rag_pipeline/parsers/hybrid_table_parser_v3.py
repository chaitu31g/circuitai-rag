"""
hybrid_table_parser_v3.py – Surgical pdfplumber Coordinate-Based Parser
======================================================================
Architecture:
  1. Docling → Region Detection (bboxes only)
  2. pdfplumber → Word-Level Coordinate Extraction {text, x0, top, x1, bottom}
  3. Spatial Clustering (Y-Axis) → Row Formation
  4. Header Anchor Discovery (X-Axis) → Column Mapping
  5. Semantic Multi-Line Merging + Context Forward-Fill
  6. Final Structuring (Parameter, Symbol, Condition, Value, Unit)
"""

import re
import logging
import pdfplumber
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from ingestion.datasheet_chunker import Chunk

logger = logging.getLogger(__name__)

# --- CONFIGURATION (Coordinate Anchors & Symbols) ---
_HEADER_SYMBOLS = ["parameter", "symbol", "condition", "value", "unit"]

# Clean symbols specifically for semiconductor ratings
_SYMBOL_FIXES = {
    "I D": "ID", "V GS": "VGS", "V DS": "VDS", "T A": "TA", "T j": "Tj", 
    "V (BR)DSS": "V(BR)DSS", "RDS (on)": "RDS(on)", "C iss": "Ciss", 
    "C oss": "Coss", "C rss": "Crss"
}

# ─────────────────────────────────────────────────────────────────
# 1. CORE COORDINATE-BASED EXTRACTION
# ─────────────────────────────────────────────────────────────────
def extract_surgical_table(pdf_path: str, page_no: int, bbox: List[float]) -> List[Dict]:
    """
    Extract every word with precision {x0, top, x1, bottom} using pdfplumber.
    Coordinates are standard PDF points (0,0 at bottom-left).
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_no - 1]
        # Restrict extraction to the table area detected by Docling
        # pdfplumber bbox: (x0, y0, x1, y1) -> left, top, right, bottom
        cropped = page.within_bbox((bbox[0], page.height - bbox[3], bbox[2], page.height - bbox[1]))
        
        words = cropped.extract_words(keep_blank_chars=False, x_tolerance=3, y_tolerance=3)
        return sorted(words, key=lambda w: (w["top"], w["x0"]))


# ─────────────────────────────────────────────────────────────────
# 2. ROW GROUPING (Y-Clustering)
# ─────────────────────────────────────────────────────────────────
def group_words_into_rows(words: List[Dict], row_tolerance: float = 2.0) -> List[List[Dict]]:
    """
    Group words within a certain Y-offset into the same Row.
    """
    if not words: return []
    
    rows = []
    current_row = [words[0]]
    last_top = words[0]["top"]
    
    for w in words[1:]:
        if abs(w["top"] - last_top) <= row_tolerance:
            current_row.append(w)
        else:
            rows.append(sorted(current_row, key=lambda x: x["x0"]))
            current_row = [w]
            last_top = w["top"]
    
    if current_row:
        rows.append(sorted(current_row, key=lambda x: x["x0"]))
    return rows


# ─────────────────────────────────────────────────────────────────
# 3. COLUMN BOUNDARY DETECTION (H-Zoning)
# ─────────────────────────────────────────────────────────────────
def detect_column_anchors(rows: List[List[Dict]]) -> Dict[str, Tuple[float, float]]:
    """
    Scan top rows to find where Parameter, Symbol, Condition, Value, Unit start/end.
    Returns {semantic_name: (x0, x1)}
    """
    anchors = {}
    for row in rows[:3]: # Scan top 3 rows for headers
        for w in row:
            txt = w["text"].lower().strip()
            for h in _HEADER_SYMBOLS:
                if h in txt and h not in anchors:
                    anchors[h] = (w["x0"] - 5, w["x1"] + 5)
    
    # Auto-adjust gaps based on discovered headers
    if "parameter" in anchors and "symbol" in anchors:
        # If symbol is to the right of parameter, let parameter expand until symbol
        anchors["parameter"] = (anchors["parameter"][0], anchors["symbol"][0] - 2)
        
    return anchors


# ─────────────────────────────────────────────────────────────────
# 4. DATA MAPPING & MULTI-LINE MERGE
# ─────────────────────────────────────────────────────────────────
def process_rows_surgical(rows: List[List[Dict]], anchors: Dict[str, Tuple[float, float]]) -> List[Dict]:
    """
    Map every row to a structured dictionary and merge multi-line entries.
    """
    structured_data = []
    last_processed_row = None
    
    # Default zone fallback if anchors weren't found clearly
    # Normalized for a standard 5-column datasheet table
    zones = {
        "parameter": anchors.get("parameter", (0, 150)),
        "symbol":    anchors.get("symbol",    (150, 200)),
        "condition": anchors.get("condition", (200, 350)),
        "value":     anchors.get("value",     (350, 480)),
        "unit":      anchors.get("unit",      (480, 550))
    }

    last_parameter_str = ""
    last_symbol_str = ""
    last_unit_str = ""
    
    for row_words in rows:
        row_content = {"parameter": "", "symbol": "", "condition": "", "value": "", "unit": ""}
        row_has_data = False
        
        for w in row_words:
            center_x = (w["x0"] + w["x1"]) / 2
            # Assign word to a zone
            found_zone = False
            for zone_name, (z0, z1) in zones.items():
                if z0 <= center_x <= z1:
                    row_content[zone_name] = (row_content[zone_name] + " " + w["text"]).strip()
                    found_zone = True
                    break
            
            # If word is way out of bounds, attach to nearest zone
            if not found_zone:
                 if center_x < zones["parameter"][1]: row_content["parameter"] += " " + w["text"]
                 elif center_x > zones["unit"][0]:    row_content["unit"] += " " + w["text"]

        # CLEAN SYMBOLS & UNITS
        row_content["parameter"] = row_content["parameter"].strip()
        row_content["symbol"]    = _clean_engine_text(row_content["symbol"])
        row_content["condition"] = _clean_engine_text(row_content["condition"])
        row_content["value"]     = row_content["value"].strip()
        row_content["unit"]      = row_content["unit"].strip()

        # REJECT HEADER ROWS
        if any(h in row_content["parameter"].lower() for h in _HEADER_SYMBOLS):
            continue

        # MULTI-LINE PARAMETER MERGE
        # If this row ONLY has a parameter name (no value), it's a split line
        if row_content["parameter"] and not row_content["value"] and not row_content["condition"]:
            last_parameter_str = (last_parameter_str + " " + row_content["parameter"]).strip()
            continue
        
        # FORWARD-FILL MERGED CELLS
        if not row_content["parameter"]: row_content["parameter"] = last_parameter_str
        else: last_parameter_str = row_content["parameter"]
        
        if not row_content["symbol"]:    row_content["symbol"] = last_symbol_str
        else: last_symbol_str = row_content["symbol"]
        
        if not row_content["unit"]:      row_content["unit"] = last_unit_str
        else: last_unit_str = row_content["unit"]

        # Only accept rows with a value or condition
        if any(ch.isdigit() for ch in row_content["value"]) or row_content["condition"]:
            structured_data.append(row_content.copy())

    return structured_data


def _clean_engine_text(text: str) -> str:
    """Fix common semiconductor OCR artifacts."""
    for bad, good in _SYMBOL_FIXES.items():
        text = text.replace(bad, good)
    # Join isolated characters in symbols ID, VGS, etc.
    text = re.sub(r'\b([A-Z])\s+([A-Z])\b', r'\1\2', text)
    return text.strip() or "-"


# ─────────────────────────────────────────────────────────────────
# 5. MAIN ENTRYPOINT
# ─────────────────────────────────────────────────────────────────
def extract_tables_hybrid_v3(
    pdf_path:     str,
    docling_data: dict,
    part_number:  str,
) -> List[Chunk]:
    """
    Primary Entry Point: Strategic Coordinate-Based Parsing.
    """
    all_chunks: List[Chunk] = []
    
    # 1. Get Table regions from Docling
    tables = docling_data.get("tables", [])
    
    for i, tbl in enumerate(tables):
        prov = (tbl.get("prov") or [{}])[0]
        page_no = prov.get("page_no")
        bbox    = prov.get("bbox")
        if not page_no or not bbox: continue
        
        # Convert dictionary bbox to list if needed
        if isinstance(bbox, dict):
            bbox_list = [bbox.get("l",0), bbox.get("t",0), bbox.get("r",0), bbox.get("b",0)]
        else:
            bbox_list = list(bbox[:4])

        try:
            # 2. Extract words with coords
            words = extract_surgical_table(pdf_path, page_no, bbox_list)
            
            # 3. Form Rows
            grouped_rows = group_words_into_rows(words)
            
            # 4. Map Columns
            anchors = detect_column_anchors(grouped_rows)
            
            # 5. Process & Structure
            final_data = process_rows_surgical(grouped_rows, anchors)
            
            # 6. Convert to RAG Chunks
            for record in final_data:
                # Deduplicate parameter name if it merged redundantly
                param = record["parameter"].replace("Continuous drain current Continuous drain current", "Continuous drain current")
                
                # Format text exactly for RAG
                chunk_text = (
                    f"Parameter: {param}\n"
                    f"Symbol: {record['symbol']}\n"
                    f"Condition: {record['condition']}\n"
                    f"Value: {record['value']}\n"
                    f"Unit: {record['unit']}"
                )
                
                all_chunks.append(Chunk(
                    text=chunk_text,
                    chunk_type="parameter_row",
                    metadata={
                        "part_number": part_number,
                        "page": page_no,
                        "parameter": param,
                        "table_index": i
                    }
                ))
            
            logger.info(f"Surgically extracted {len(final_data)} rows from Table {i+1} on page {page_no}.")

        except Exception as e:
            logger.error(f"Surgical extraction failed for Table {i}: {e}")

    return all_chunks
