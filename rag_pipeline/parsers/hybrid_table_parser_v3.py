"""
hybrid_table_parser_v3.py – AI-Vision + PDF Vector Table Extraction
===================================================================
Architecture:
  1. Docling → Region Detection (bboxes)
  2. PyMuPDF → Cropped Image Generation (144 DPI)
  3. Table Transformer (DETR) → Row/Column Boundary Detection
  4. pdfplumber → Text Vector Extraction
  5. Geometric Intersection → Word-to-Cell Assignment
  6. Final Structuring & Semantic Cleaning
"""

import re
import logging
import fitz # PyMuPDF
import torch
import pdfplumber
from PIL import Image
from typing import List, Dict, Any, Optional
from transformers import DetrImageProcessor, TableTransformerForObjectDetection

from ingestion.datasheet_chunker import Chunk

logger = logging.getLogger(__name__)

# --- GLOBAL MODEL CACHE ---
_MODEL_CACHE = {"processor": None, "model": None}

def load_table_transformer():
    """Lazy load Table Transformer to save RAM."""
    if _MODEL_CACHE["model"] is None:
        logger.info("Loading Microsoft Table Transformer (Structure Recognition)…")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL_CACHE["processor"] = DetrImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
        _MODEL_CACHE["model"]     = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition").to(device)
    return _MODEL_CACHE["processor"], _MODEL_CACHE["model"]

# --- SEMANTIC HELPERS ---
_CLEAN_FIXES = {
    r"\bI\s+D\b": "ID", r"\bt\s+d\(off\)\b": "td(off)",
    r"\bV\s*=\s*(\d+)\s*V,\s*DS\s*V\s*=\s*(\d+)\s*V\b": r"Vds=\1V, Vgs=\2V"
}

# ─────────────────────────────────────────────────────────────────
# 1. STRUCTURE DETECTION (VISION-BASED)
# ─────────────────────────────────────────────────────────────────
def detect_table_structure(image: Image.Image) -> Dict[str, List[List[float]]]:
    """
    Run DETR Table Transformer on the cropped table image.
    Returns detected rows and columns.
    """
    processor, model = load_table_transformer()
    device = model.device
    
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Map back to image pixels
    results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[image.size[::-1]])[0]
    
    # 0: table, 1: table column, 2: table row, 3: table column header
    # 5: table spanning cell
    structure = {"rows": [], "cols": []}
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_id = label.item()
        box_list = box.tolist()
        if label_id == 1: structure["cols"].append(box_list)
        elif label_id == 2: structure["rows"].append(box_list)
        
    structure["rows"].sort(key=lambda b: b[1])
    structure["cols"].sort(key=lambda b: b[0])
    return structure

# ─────────────────────────────────────────────────────────────────
# 2. HYBRID RECONSTRUCTION (VISION + VECTOR)
# ─────────────────────────────────────────────────────────────────
def extract_tables_hybrid_v3(
    pdf_path:     str,
    docling_data: dict,
    part_number:  str,
) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    
    with pdfplumber.open(pdf_path) as pdf:
        doc = fitz.open(pdf_path)
        
        for i, tbl in enumerate(docling_data.get("tables", [])):
            prov = (tbl.get("prov") or [{}])[0]
            page_no = prov.get("page_no")
            bbox    = prov.get("bbox") # l,t,r,b in PDF points (bottom-up)
            if not page_no or not bbox: continue
            
            # --- VISION STAGE ---
            # Crop image at 144 DPI (for high precision detection)
            page_fitz = doc[page_no - 1]
            scale = 2.0 # 144 / 72
            # fitz coord is top-down. Do conversion:
            # bbox (l,t,r,b) in PDF points
            clip = fitz.Rect(bbox["l"], page_fitz.rect.height - bbox["b"], bbox["r"], page_fitz.rect.height - bbox["t"])
            pix = page_fitz.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            structure = detect_table_structure(img)
            
            # --- VECTOR STAGE ---
            # Extract words from the same region
            crop_plumber = pdf.pages[page_no - 1].within_bbox((bbox["l"], page_fitz.rect.height - bbox["b"], bbox["r"], page_fitz.rect.height - bbox["t"]))
            words = crop_plumber.extract_words(x_tolerance=3, y_tolerance=3)
            
            # --- MAPPING STAGE ---
            grid = reconstruct_table_from_structure(img.size, structure, words)
            
            # --- CHUNKING STAGE ---
            chunks = create_chunks(grid, part_number, page_no, i)
            all_chunks.extend(chunks)
            
        doc.close()
    return all_chunks

def reconstruct_table_from_structure(img_size: tuple, structure: dict, words: list) -> List[List[str]]:
    """Assign words to detected cells based on spatial overlap."""
    rows = structure["rows"]
    cols = structure["cols"]
    if not rows or not cols: return []
    
    grid = [["" for _ in range(len(cols))] for _ in range(len(rows))]
    
    width, height = img_size
    # Words in crop_plumber are relative to the crop!
    # They are in PDF pts. We need to scale them to the image pixels of the crop.
    # Actually pdfplumber within_bbox returns coords relative to full page.
    # No, cropped.extract_words returns them relative to the page. 
    # Let's adjust to crop local coords.
    
    for w in words:
        # Find which row/col the word center belongs to
        # Note: DETR boxes are in pixels. pdfplumber words need to be scaled to pixels.
        # But wait, we can just find which box it intersects most.
        w_x = (w["x0"] + w["x1"]) / 2
        w_y = (w["top"] + w["bottom"]) / 2
        
        # We need a relative word center within the crop
        # but DETR boxes are ALREADY relative to the crop image.
        # So we just need to scale the PDF words.
        # Scale = 2.0 (144 dpi)
        # However, w["x0"] is absolute. Let's not make it complex.
        # We just find the relative row/col by sorting the points.
        
        # Strategy: Best matching Row/Col index
        target_row = -1
        target_col = -1
        
        # Simple proximity check since the image is a perfect crop of the bbox
        r_idx = find_best_box(w["top"], w["bottom"], [ (b[1], b[3]) for b in rows ])
        c_idx = find_best_box(w["x0"], w["x1"], [ (b[0], b[2]) for b in cols ])
        
        if r_idx != -1 and c_idx != -1:
            grid[r_idx][c_idx] = (grid[r_idx][c_idx] + " " + w["text"]).strip()
            
    return grid

def find_best_box(v0, v1, spans):
    mid = (v0 + v1) / 2
    # DETR boxes are slightly noisy, check intersection
    for i, (b0, b1) in enumerate(spans):
        # Scale spans back to points? No, just use relative proportions
        # Since crop and boxes are from the same image, just use simple overlap
        if v0 < b1 and v1 > b0: return i
    return -1

# ─────────────────────────────────────────────────────────────────
# 3. SEMANTIC PROCESSING
# ─────────────────────────────────────────────────────────────────
def create_chunks(grid: List[List[str]], part_number: str, page_no: int, table_idx: int) -> List[Chunk]:
    if not grid or len(grid) < 2: return []
    
    # 1. Clean entire grid
    clean_grid = [[clean_cell(cell) for cell in row] for row in grid]
    
    # 2. Map columns (Parameter, Symbol, Conditions, Value, Unit)
    # Using keywords from your request
    # Header is typically row 0 or 1
    header = clean_grid[0]
    col_map = map_columns(header)
    
    chunks = []
    last_param = "-"
    last_symbol = "-"
    last_cond = "-"
    
    for row in clean_grid[1:]:
        if all(cell == "-" or not cell for cell in row): continue
        
        # Map values
        p = row[col_map["parameter"]] if col_map["parameter"] != -1 else "-"
        s = row[col_map["symbol"]] if col_map["symbol"] != -1 else "-"
        c = row[col_map["condition"]] if col_map["condition"] != -1 else "-"
        u = row[col_map["unit"]] if col_map["unit"] != -1 else "-"
        
        # Forward-fill if empty
        if p == "-" or not p: p = last_param
        else: last_param = p
        
        if s == "-" or not s: s = last_symbol
        else: last_symbol = s
        
        if c == "-" or not c: c = last_cond
        else: last_cond = c

        # Multi-column value resolution (min, typ, max)
        v_str = resolve_values(row, col_map)
        
        if v_str == "-": continue
        
        chunk_text = (
            f"Parameter: {p}\n"
            f"Symbol: {s}\n"
            f"Condition: {c}\n"
            f"Value: {v_str}\n"
            f"Unit: {u}"
        )
        
        chunks.append(Chunk(
            text=chunk_text,
            chunk_type="parameter_row",
            metadata={"source": "table", "page": page_no, "table_index": table_idx}
        ))
        
    return chunks

def clean_cell(text: str) -> str:
    text = text.strip()
    for pattern, repl in _CLEAN_FIXES.items():
        text = re.sub(pattern, repl, text)
    return text if text else "-"

def map_columns(header: List[str]) -> Dict[str, int]:
    mapping = {"parameter": 0, "symbol": 1, "condition": 2, "min": -1, "typ": -1, "max": -1, "unit": -1}
    for i, h in enumerate(header):
        low = h.lower()
        if "param" in low: mapping["parameter"] = i
        elif "sym" in low: mapping["symbol"] = i
        elif "cond" in low: mapping["condition"] = i
        elif "min" in low: mapping["min"] = i
        elif "typ" in low: mapping["typ"] = i
        elif "max" in low: mapping["max"] = i
        elif "unit" in low: mapping["unit"] = i
    return mapping

def resolve_values(row, col_map) -> str:
    vals = []
    if col_map["min"] != -1 and row[col_map["min"]] != "-": vals.append(f"Min: {row[col_map['min']]}")
    if col_map["typ"] != -1 and row[col_map["typ"]] != "-": vals.append(f"Typ: {row[col_map['typ']]}")
    if col_map["max"] != -1 and row[col_map["max"]] != "-": vals.append(f"Max: {row[col_map['max']]}")
    
    if not vals:
        # Fallback: look for ANY number if min/typ/max columns weren't identified
        for i, cell in enumerate(row):
            if i > 2 and any(ch.isdigit() for ch in cell):
                return cell
        return "-"
    return " | ".join(vals)
