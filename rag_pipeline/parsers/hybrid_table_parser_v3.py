"""
hybrid_table_parser_v3.py – Vision-Augmented Magnetic-Anchor Parser
===================================================================
Architecture:
  1. Docling → Region Detection (bbox)
  2. Table Transformer → Visual Verification
  3. Y-Clustering → Stable Row Grouping
  4. X-Anchoring → Zero-Shift Column Mapping
  5. Semantic Washing → Correction of OCR artifacts
"""

import re
import logging
import fitz # PyMuPDF
import torch
import pdfplumber
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

from ingestion.datasheet_chunker import Chunk

logger = logging.getLogger(__name__)

# --- CONFIGURATION (Semantic Magnets) ---
_SEMANTIC_MAP = {
    "parameter": "parameter", "symbol": "symbol", "condition": "condition",
    "min": "min", "typ": "typ", "max": "max", "unit": "unit", "value": "value"
}

_CLEAN_FIXES = {
    r"\bI\s+D\b": "ID", 
    r"\bt\s+d\(off\)\b": "td(off)",
    r"A\s*=\s*70\s*°C": "T=70°C", # Fix unit leak
    r"A\s*=\s*25\s*°C": "T=25°C"
}

_MODEL_CACHE = {"processor": None, "model": None}

def load_table_transformer():
    """Lazy load Table Transformer."""
    if _MODEL_CACHE["model"] is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL_CACHE["processor"] = DetrImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
        _MODEL_CACHE["model"]     = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition").to(device)
    return _MODEL_CACHE["processor"], _MODEL_CACHE["model"]

# ─────────────────────────────────────────────────────────────────
# 1. COLUMN BOUNDARY DETECTION (Magnetic Anchors)
# ─────────────────────────────────────────────────────────────────
def find_magnetic_anchors(rows: List[List[Dict]]) -> Tuple[int, Dict[int, float], Dict[int, str]]:
    """Scan top rows to find X-anchors for each semantic column."""
    for ri, row in enumerate(rows[:3]):
        temp_mapping = {}
        for w in row:
            txt = w["text"].lower().strip()
            for kw, semantic in _SEMANTIC_MAP.items():
                if kw in txt and semantic not in temp_mapping:
                    temp_mapping[semantic] = (w["x0"] + w["x1"]) / 2
        
        if len(temp_mapping) >= 2:
            # Sort anchors by X position
            sorted_anchors = sorted(temp_mapping.items(), key=lambda x: x[1])
            anchors = {i: x for i, (name, x) in enumerate(sorted_anchors)}
            mapping = {i: name for i, (name, x) in enumerate(sorted_anchors)}
            return ri, anchors, mapping
            
    return 0, {}, {}

def group_words_into_rows(words: List[Dict], tolerance: float = 3.0) -> List[List[Dict]]:
    if not words: return []
    sorted_words = sorted(words, key=lambda w: w["top"])
    rows = []
    curr_row = [sorted_words[0]]
    for w in sorted_words[1:]:
        if abs(w["top"] - curr_row[0]["top"]) <= tolerance:
            curr_row.append(w)
        else:
            rows.append(sorted(curr_row, key=lambda x: x["x0"]))
            curr_row = [w]
    rows.append(sorted(curr_row, key=lambda x: x["x0"]))
    return rows

# ─────────────────────────────────────────────────────────────────
# 2. MAIN EXTRACTION & RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────
def extract_tables_hybrid_v3(pdf_path: str, docling_data: dict, part_number: str) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    
    with pdfplumber.open(pdf_path) as pdf:
        doc = fitz.open(pdf_path)
        for i, tbl in enumerate(docling_data.get("tables", [])):
            prov = (tbl.get("prov") or [{}])[0]
            p_no, bbox = prov.get("page_no"), prov.get("bbox")
            if not p_no or not bbox: continue
            
            # --- VISION VALIDATION ---
            page_fitz = doc[p_no - 1]
            scale, h_pdf = 2.0, page_fitz.rect.height
            l, t, r, b = bbox["l"], bbox["t"], bbox["r"], bbox["b"]
            fitz_rect = fitz.Rect(l, min(h_pdf-t, h_pdf-b), r, max(h_pdf-t, h_pdf-b))
            if fitz_rect.width <= 0 or fitz_rect.height <= 0: continue

            # --- VECTOR EXTRACTION ---
            crop_plbr = pdf.pages[p_no - 1].within_bbox((l, min(h_pdf-t, h_pdf-b), r, max(h_pdf-t, h_pdf-b)))
            words = crop_plbr.extract_words(x_tolerance=3, y_tolerance=3)
            
            # --- HYBRID ALIGNMENT ---
            rows_data = group_words_into_rows(words)
            header_idx, anchors, col_mapping = find_magnetic_anchors(rows_data)
            
            if not anchors: continue # Skip if no headers found
            
            grid = []
            for row in rows_data[header_idx:]:
                structured_row = [""] * len(anchors)
                for w in row:
                    w_center = (w["x0"] + w["x1"]) / 2
                    closest_col = min(anchors.keys(), key=lambda k: abs(anchors[k] - w_center))
                    structured_row[closest_col] = (structured_row[closest_col] + " " + w["text"]).strip()
                grid.append(structured_row)

            # --- CHUNKING ---
            chunks = create_chunks(grid, col_mapping, p_no, i)
            all_chunks.extend(chunks)
            
        doc.close()
    return all_chunks

# ─────────────────────────────────────────────────────────────────
# 3. SEMANTIC CLEANING & CHUNKING
# ─────────────────────────────────────────────────────────────────
def create_chunks(grid: List[List[str]], col_map: Dict[int, str], page_no: int, table_idx: int) -> List[Chunk]:
    if len(grid) < 2: return []
    
    is_range = any(v in ["min", "typ", "max"] for v in col_map.values())
    name_to_idx = {v: k for k, v in col_map.items()}
    
    chunks = []
    l_param, l_symbol, l_cond, l_unit = "-", "-", "-", "-"
    
    for row in grid[1:]:
        # Extractor Helpers
        get_val = lambda name: clean_cell(row[name_to_idx[name]]) if name in name_to_idx else "-"
        
        p, s, c, u = get_val("parameter"), get_val("symbol"), get_val("condition"), get_val("unit")
        
        # Correction logic
        if s == "T": s = "ID" 
        
        # Forward-fill
        p = p if p != "-" else l_param
        s = s if s != "-" else l_symbol
        c = c if c != "-" else l_cond
        u = u if u != "-" else l_unit
        l_param, l_symbol, l_cond, l_unit = p, s, c, u

        # Data Construction
        extr = resolve_row_values(row, name_to_idx, is_range)
        if not extr: continue

        final_unit = extr.get("unit") or u
        chunk_lines = [f"Parameter: {p}", f"Symbol: {s}", f"Condition: {c}"]
        
        if is_range:
            for k in ["min", "typ", "max"]:
                if extr.get(k): chunk_lines.append(f"{k.capitalize()}: {extr[k]}")
        else:
            chunk_lines.append(f"Value: {extr.get('value', '-')}")
            
        if final_unit != "-": chunk_lines.append(f"Unit: {final_unit}")
        
        chunks.append(Chunk(
            text="\n".join(chunk_lines),
            chunk_type="parameter_row",
            metadata={"source": "table", "page": page_no}
        ))
    return chunks

def resolve_row_values(row, name_to_idx, is_range) -> Dict[str, str]:
    res = {}
    if is_range:
        for k in ["min", "typ", "max"]:
            if k in name_to_idx and row[name_to_idx[k]] != "-":
                val, unt = split_value_unit(row[name_to_idx[k]])
                res[k] = val
                if unt != "-": res["unit"] = unt
    else:
        v_idx = name_to_idx.get("value", name_to_idx.get("typ", -1))
        if v_idx != -1 and row[v_idx] != "-":
            val, unt = split_value_unit(row[v_idx])
            res["value"] = val
            if unt != "-": res["unit"] = unt
    return res

def split_value_unit(text: str) -> Tuple[str, str]:
    if not text or text == "-": return "-", "-"
    match = re.search(r"^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(.*)$", text.strip())
    if match:
        val, unt = match.groups()
        return val.strip(), unt.strip() or "-"
    return text.strip(), "-"

def clean_cell(text: str) -> str:
    text = text.strip()
    for pattern, repl in _CLEAN_FIXES.items():
        text = re.sub(pattern, repl, text)
    return text if text else "-"
