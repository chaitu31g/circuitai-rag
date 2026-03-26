"""
hybrid_table_parser_v3.py – Docling + Table Transformer + pdfplumber Pipeline
==============================================================================
Architecture:
  PDF → Docling (section/bbox detection)
      → PyMuPDF (crop table images)
      → Microsoft Table Transformer (row/col structure)
      → pdfplumber (word coordinates)
      → Cell assignment (overlap matching)
      → Column mapping + cleaning
      → Chunk generation
      → Fallback: pdfplumber-only if Table Transformer fails
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

from ingestion.datasheet_chunker import _get_table_contexts, _detect_section, _SKIP_TYPES, Chunk

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# MODEL LOADING  (lazy singleton – loaded once, shared everywhere)
# ─────────────────────────────────────────────────────────────────
_processor: Optional[AutoImageProcessor] = None
_model: Optional[TableTransformerForObjectDetection] = None

def _get_model() -> Tuple[AutoImageProcessor, TableTransformerForObjectDetection]:
    global _processor, _model
    if _model is None:
        logger.info("Loading Microsoft Table Transformer (structure recognition)…")
        _processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        _model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        _model.eval()
        logger.info("Table Transformer loaded ✓")
    return _processor, _model

# Label indices from Table Transformer's id2label
_TABLE_LABELS = {
    "table row": 2,
    "table column": 3,
    "table column header": 4,
}

# ─────────────────────────────────────────────────────────────────
# 1. TABLE REGION EXTRACTION  (Docling → bbox + section)
# ─────────────────────────────────────────────────────────────────
def extract_table_regions(docling_data: dict) -> List[Dict[str, Any]]:
    """Return one region dict per Docling-detected table."""
    tables = docling_data.get("tables", [])
    texts  = docling_data.get("texts",  [])
    regions: List[Dict[str, Any]] = []

    for i, tbl in enumerate(tables):
        prov  = (tbl.get("prov") or [{}])[0]
        page_no = prov.get("page_no")
        bbox    = prov.get("bbox")
        if not page_no or not bbox:
            continue

        # Normalise bbox → [l, t, r, b]  (handle Docling dict OR list format)
        if isinstance(bbox, dict):
            valid_bbox = [bbox.get("l",0), bbox.get("t",0), bbox.get("r",0), bbox.get("b",0)]
            bottom_left_origin = True
        else:
            valid_bbox = list(bbox[:4])
            bottom_left_origin = False

        # Infer section from nearest Docling text block on the same page
        best_sec = "electrical_characteristics"
        for t in texts:
            t_page = (t.get("prov") or [{}])[0].get("page_no")
            if t_page == page_no:
                d = _detect_section(t.get("text", ""))
                if d and d not in _SKIP_TYPES:
                    best_sec = d
                    break

        regions.append({
            "table_index":          i,
            "page_no":              page_no,
            "bbox":                 valid_bbox,
            "bottom_left_origin":   bottom_left_origin,
            "docling_tbl":          tbl,
            "section":              best_sec,
        })

    return regions


# ─────────────────────────────────────────────────────────────────
# 2. IMAGE CROPPING  (PyMuPDF → PIL Image)
# ─────────────────────────────────────────────────────────────────
def crop_table_images(pdf_path: str, regions: List[Dict[str, Any]], dpi: int = 150) -> List[Dict[str, Any]]:
    """
    Attach a PIL Image to every region dict.
    `dpi=150` is fast enough for structure detection (no raw OCR needed).
    """
    doc = fitz.open(pdf_path)

    for region in regions:
        page_num = region["page_no"] - 1
        if page_num < 0 or page_num >= doc.page_count:
            continue

        page = doc.load_page(page_num)
        l, t, r, b = region["bbox"]

        # Invert Y if Docling used bottom-left origin
        ph = page.rect.height
        if region["bottom_left_origin"] and t > b:
            y0, y1 = ph - t, ph - b
        else:
            y0, y1 = t, b

        # Apply padding
        rect = fitz.Rect(max(0, l-5), max(0, y0-5),
                         min(page.rect.width, r+5), min(ph, y1+5))
        pix  = page.get_pixmap(clip=rect, dpi=dpi)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        region["pil_image"]   = img
        region["page_rect"]   = rect          # fitz.Rect of the crop in PDF coords
        region["page_height"] = ph

    doc.close()
    return regions


# ─────────────────────────────────────────────────────────────────
# 3. TABLE STRUCTURE DETECTION  (Table Transformer)
# ─────────────────────────────────────────────────────────────────
def detect_table_structure(image: Image.Image, threshold: float = 0.5) -> Dict[str, List]:
    """
    Run Table Transformer on a PIL image.
    Returns {"rows": [...], "columns": [...]} where each entry is a
    normalised [x0,y0,x1,y1] box in IMAGE pixel space.
    """
    processor, model = _get_model()
    inputs  = processor(images=image, return_tensors="pt")
    W, H    = image.size

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=[(H, W)]
    )[0]

    rows, columns = [], []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        lbl_name = model.config.id2label[label.item()]
        box_px   = [round(v, 1) for v in box.tolist()]
        if "row" in lbl_name:
            rows.append(box_px)
        elif "column" in lbl_name:
            columns.append(box_px)

    # Sort top-to-bottom and left-to-right
    rows    = sorted(rows,    key=lambda b: b[1])
    columns = sorted(columns, key=lambda b: b[0])
    return {"rows": rows, "columns": columns}


# ─────────────────────────────────────────────────────────────────
# 4. TEXT EXTRACTION  (pdfplumber word coordinates)
# ─────────────────────────────────────────────────────────────────
def extract_words_pdfplumber(pdf_path: str, page_no: int, crop_rect: fitz.Rect) -> List[Dict]:
    """
    Extract word-level bounding boxes from pdfplumber for the table crop region.
    Coordinates are in PDF-space (same unit as fitz.Rect).
    """
    words = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_no - 1]
        # Restrict pdfplumber to the crop area
        cropped = page.within_bbox((
            crop_rect.x0, crop_rect.y0,
            crop_rect.x1, crop_rect.y1
        ))
        for w in (cropped.extract_words() or []):
            words.append({
                "text":   w["text"],
                "x0":     w["x0"] - crop_rect.x0,   # shift to image-local coords
                "x1":     w["x1"] - crop_rect.x0,
                "top":    w["top"] - crop_rect.y0,
                "bottom": w["bottom"] - crop_rect.y0,
            })
    return words


# ─────────────────────────────────────────────────────────────────
# 5. CELL RECONSTRUCTION (overlap matching)
# ─────────────────────────────────────────────────────────────────
def _overlap_1d(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def _overlap_area(word: Dict, row_box, col_box) -> float:
    ox = _overlap_1d(word["x0"], word["x1"], col_box[0], col_box[2])
    oy = _overlap_1d(word["top"], word["bottom"], row_box[1], row_box[3])
    return ox * oy

def assign_words_to_cells(
    words:   List[Dict],
    rows:    List[List],
    columns: List[List],
) -> List[List[str]]:
    """
    Assign each word to the (row, col) cell with maximum overlap area.
    Returns a 2-D list[row][col] of strings.
    """
    n_rows = len(rows)
    n_cols = len(columns)
    if n_rows == 0 or n_cols == 0:
        return []

    grid: List[List[List[str]]] = [[[] for _ in range(n_cols)] for _ in range(n_rows)]

    for word in words:
        best_r, best_c, best_area = 0, 0, -1.0
        for ri, row_box in enumerate(rows):
            for ci, col_box in enumerate(columns):
                area = _overlap_area(word, row_box, col_box)
                if area > best_area:
                    best_area = area
                    best_r, best_c = ri, ci
        if best_area > 0:
            grid[best_r][best_c].append(word["text"])

    return [[" ".join(cell) for cell in row] for row in grid]


def build_table_grid(words, structure) -> List[List[str]]:
    """Convenience wrapper → returns grid[row][col]."""
    return assign_words_to_cells(words, structure["rows"], structure["columns"])


# ─────────────────────────────────────────────────────────────────
# 6. HEADER DETECTION
# ─────────────────────────────────────────────────────────────────
_HEADER_KEYWORDS = {
    "parameter": "parameter",
    "symbol":    "symbol",
    "condition": "condition",
    "cond":      "condition",
    "test":      "condition",
    "min":       "min",
    "min.":      "min",
    "typ":       "typ",
    "typ.":      "typ",
    "max":       "max",
    "max.":      "max",
    "unit":      "unit",
    "units":     "unit",
    "value":     "typ",   # single-value tables
    "values":    "typ",
}

def map_columns(header_row: List[str]) -> Dict[int, str]:
    """
    Returns {col_index: semantic_name} based on header keywords.
    Unrecognised columns get "val_N" (treated as numeric later).
    """
    mapping: Dict[int, str] = {}
    val_counter = 0
    for i, h in enumerate(header_row):
        key = h.strip().lower()
        if key in _HEADER_KEYWORDS:
            mapping[i] = _HEADER_KEYWORDS[key]
        else:
            mapping[i] = f"val_{val_counter}"
            val_counter += 1
    return mapping


# ─────────────────────────────────────────────────────────────────
# 7. TEXT CLEANING
# ─────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """General OCR-noise removal for symbols and conditions."""
    if not text:
        return "-"
    text = text.strip()

    # Known OCR subscript re-attachments
    text = re.sub(r'\bC\s+oss\b',    'Coss',    text)
    text = re.sub(r'\bC\s+iss\b',    'Ciss',    text)
    text = re.sub(r'\bC\s+rss\b',    'Crss',    text)
    text = re.sub(r'\bt\s+d\(',      'td(',     text)
    text = re.sub(r'\bt\s+r\b',      'tr',      text)
    text = re.sub(r'\bt\s+f\b',      'tf',      text)
    text = re.sub(r'\bR\s+DS',       'RDS',     text)
    text = re.sub(r'\bV\s+GS',       'VGS',     text)
    text = re.sub(r'\bV\s+DS',       'VDS',     text)
    text = re.sub(r'\bI\s+D\b',      'ID',      text)

    # Condition normalisation  "V =60 V, DS …"  →  "Vds=60V"
    text = text.replace("V =60 V, DS V =0 V", "Vds=60V, Vgs=0V")
    text = re.sub(r'V\s*=\s*([\d.]+)\s*V,\s*DS', r'Vds=\1V', text)
    text = re.sub(r'V\s*=\s*([\d.]+)\s*V,\s*GS', r'Vgs=\1V', text)
    text = re.sub(r'\s*=\s*', '=', text)
    text = re.sub(r'(?<=\d)\s+([A-Za-zΩµ°]+)', r'\1', text)
    text = re.sub(r'\s+[A-Za-z]$', '', text)
    return text.strip() or "-"


def normalize_symbol(symbol: str) -> str:
    return symbol.replace(" ", "") if symbol and symbol != "-" else "-"


def extract_unit(condition: str, current_unit: str) -> Tuple[str, str]:
    unit_re = r'\s+(pF|nF|uF|F|ns|us|ms|s|MHz|kHz|Hz|mV|V|mA|uA|A|mW|W|mΩ|Ω|°C)$'
    m = re.search(unit_re, condition, re.IGNORECASE)
    if m:
        if current_unit == "-":
            current_unit = m.group(1)
        condition = condition[:m.start()].strip()
    return condition or "-", current_unit


def is_section_header_row(row: List[str]) -> bool:
    if not row:
        return False
    cells = [c.strip() for c in row]
    return (
        len(cells[0]) > 0
        and all(not c or c == "-" for c in cells[1:])
        and not any(any(ch.isdigit() for ch in c) for c in cells)
    )


# ─────────────────────────────────────────────────────────────────
# 8. ROW PROCESSING HELPERS
# ─────────────────────────────────────────────────────────────────
def _resolve_values(row: List[str], col_map: Dict[int, str]):
    """
    Pull min/typ/max either from named columns or from positional val_N columns.
    """
    named = {name: "-" for name in ["min","typ","max"]}
    val_cols = []

    for ci, name in col_map.items():
        if ci >= len(row):
            continue
        v = row[ci].strip() or "-"
        if name in ("min","typ","max"):
            named[name] = v
        elif name.startswith("val_"):
            val_cols.append(v)

    # If named columns are all "-", fall back to positional inference
    if all(v == "-" for v in named.values()):
        actives = [v for v in val_cols if v != "-"]
        if len(actives) >= 3:
            named["min"], named["typ"], named["max"] = actives[0], actives[1], actives[2]
        elif len(actives) == 2:
            named["typ"], named["max"] = actives[0], actives[1]
        elif len(actives) == 1:
            named["typ"] = actives[0]

    return named["min"], named["typ"], named["max"]


def _get_cell(row: List[str], col_map: Dict[int, str], key: str) -> str:
    for ci, name in col_map.items():
        if name == key and ci < len(row):
            return row[ci].strip() or "-"
    return "-"


# ─────────────────────────────────────────────────────────────────
# 11. CHUNK GENERATION
# ─────────────────────────────────────────────────────────────────
def create_chunks(
    grid:        List[List[str]],
    col_map:     Dict[int, str],
    section:     str,
    part_number: str,
    page_no:     int,
    table_number: int,
) -> List[Chunk]:
    """Convert a fully built grid into structured Chunk objects."""
    chunks: List[Chunk] = []

    if not grid:
        return chunks

    # Use first row as header to build the column map if not already inherited
    # (col_map already built from detect_header_row)

    last_condition = "-"
    current_section = section.replace("_", " ").title()

    for row in grid[1:]:          # skip header row
        cells = [c.strip() or "-" for c in row]
        if not cells or all(c == "-" for c in cells):
            continue
        if is_section_header_row(cells):
            current_section = cells[0]
            last_condition = "-"
            continue
        if len(cells) < 3:
            continue

        param  = _get_cell(cells, col_map, "parameter")
        if param == "-":
            param = "Unknown"
        symbol = normalize_symbol(_get_cell(cells, col_map, "symbol"))

        cond_raw = _get_cell(cells, col_map, "condition")
        if cond_raw == "-":
            condition = last_condition
        else:
            condition = clean_text(cond_raw)
            last_condition = condition

        unit = _get_cell(cells, col_map, "unit")
        condition, unit = extract_unit(condition, unit)

        min_val, typ_val, max_val = _resolve_values(cells, col_map)

        chunk_lines = [
            f"Section: {current_section}",
            f"Parameter: {param}",
            f"Symbol: {symbol}" if symbol != "-" else "",
            f"Condition: {condition}" if condition != "-" else "",
            f"Min: {min_val}",
            f"Typ: {typ_val}",
            f"Max: {max_val}",
            f"Unit: {unit}" if unit != "-" else "",
        ]
        row_txt = "\n".join(ln for ln in chunk_lines if ln)

        chunks.append(Chunk(
            text=row_txt,
            chunk_type="parameter_row",
            metadata={
                "type":         "table_row",
                "chunk_type":   "parameter_row",
                "part_number":  part_number,
                "section":      section,
                "table_number": table_number,
                "page":         page_no,
                "parameter":    param,
            }
        ))

    return chunks


# ─────────────────────────────────────────────────────────────────
# 12. FALLBACK  (pdfplumber-only extraction)
# ─────────────────────────────────────────────────────────────────
def _unmerge_multiline_row(row):
    """Expand pdfplumber rows with multi-line cells. Protects Param/Symbol/Unit columns."""
    cells = [str(c).strip() if c is not None else "" for c in row]
    value_lines = [len(c.split('\n')) for c in cells[3:-1] if c]
    max_val_lines = max(value_lines) if value_lines else 1
    if max_val_lines <= 1:
        return [[c.replace('\n', ' ').replace('  ', ' ').strip() for c in cells]]
    expanded = []
    for i in range(max_val_lines):
        new_row = []
        for col_idx, c in enumerate(cells):
            lines = [l.strip() for l in c.split('\n')]
            if col_idx in (0, 1, len(cells) - 1):
                new_row.append(" ".join(l for l in lines if l) if i == 0 else "")
            else:
                if len(lines) == max_val_lines:
                    new_row.append(lines[i])
                else:
                    new_row.append(" ".join(l for l in lines if l) if i == 0 else "")
        expanded.append(new_row)
    return expanded


def _pdfplumber_fallback(pdf_path, region, part_number, table_number) -> List[Chunk]:
    """
    Fallback when Table Transformer fails: extract table directly with pdfplumber
    using line-based grid detection.
    """
    from rag_pipeline.utils.parameter_extractor import _scrub_and_ffill

    all_chunks: List[Chunk] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[region["page_no"] - 1]
            pr = region.get("page_rect")
            cropped = page.within_bbox((pr.x0, pr.y0, pr.x1, pr.y1)) if pr else page

            for pt in cropped.find_tables():
                data = pt.extract()
                if not data or len(data) < 2:
                    continue
                headers = [str(c).replace('\n', ' ').strip() if c else "" for c in data[0]]
                clean_rows = []
                for row in data[1:]:
                    clean_rows.extend(_unmerge_multiline_row(row))
                rows_to_process = _scrub_and_ffill(clean_rows, headers)

                last_condition = "-"
                current_section = region["section"].replace("_", " ").title()
                for r in rows_to_process:
                    row = [str(x).strip() if x is not None and str(x).strip() != "" else "-" for x in r]
                    if not row or all(x == "-" for x in row):
                        continue
                    if is_section_header_row(row):
                        current_section = row[0]
                        last_condition = "-"
                        continue
                    if len(row) < 4:
                        continue
                    param = row[0] if row[0] != "-" else "Unknown"
                    symbol = normalize_symbol(row[1])
                    condition = row[2]
                    if condition == "-":
                        condition = last_condition
                    else:
                        condition = clean_text(condition)
                        last_condition = condition
                    unit = row[-1]
                    condition, unit = extract_unit(condition, unit)
                    active_vals = [v for v in row[3:-1] if v != "-"]
                    min_val, typ_val, max_val = "-", "-", "-"
                    if len(active_vals) >= 3:
                        min_val, typ_val, max_val = active_vals[0], active_vals[1], active_vals[2]
                    elif len(active_vals) == 2:
                        typ_val, max_val = active_vals[0], active_vals[1]
                    elif len(active_vals) == 1:
                        typ_val = active_vals[0]
                    row_txt = "\n".join(ln for ln in [
                        f"Section: {current_section}",
                        f"Parameter: {param}",
                        f"Symbol: {symbol}" if symbol != "-" else "",
                        f"Condition: {condition}" if condition != "-" else "",
                        f"Min: {min_val}", f"Typ: {typ_val}", f"Max: {max_val}",
                        f"Unit: {unit}" if unit != "-" else "",
                    ] if ln)
                    all_chunks.append(Chunk(
                        text=row_txt,
                        chunk_type="parameter_row",
                        metadata={
                            "type": "table_row", "chunk_type": "parameter_row",
                            "part_number": part_number, "section": region["section"],
                            "table_number": table_number, "page": region["page_no"],
                            "parameter": param,
                        }
                    ))
    except Exception as e:
        logger.warning(f"Fallback extraction failed for table {table_number}: {e}")
    return all_chunks



# ─────────────────────────────────────────────────────────────────
# MAIN ENTRYPOINT
# ─────────────────────────────────────────────────────────────────
def extract_tables_hybrid_v3(
    pdf_path:     str,
    docling_data: dict,
    part_number:  str,
) -> List[Chunk]:
    """
    Full pipeline: Docling → crop → Table Transformer → pdfplumber words
                → cell grid → column map → chunks.
    Falls back to pdfplumber-only on any Table Transformer failure.
    """
    all_chunks: List[Chunk] = []
    regions = extract_table_regions(docling_data)
    regions = crop_table_images(pdf_path, regions)

    for region in regions:
        table_num = region["table_index"] + 1

        if "pil_image" not in region:
            logger.warning(f"No image for table {table_num}. Skipping.")
            continue

        try:
            # ── Structure detection ──────────────────────────────────
            structure = detect_table_structure(region["pil_image"])

            if not structure["rows"] or not structure["columns"]:
                raise ValueError("No rows or columns detected by Table Transformer.")

            # ── Word extraction (pdfplumber, PDF coords) ─────────────
            words = extract_words_pdfplumber(
                pdf_path, region["page_no"], region["page_rect"]
            )

            # ── Cell assignment ──────────────────────────────────────
            grid = build_table_grid(words, structure)

            if not grid:
                raise ValueError("Empty grid after cell assignment.")

            # ── Header → column map ──────────────────────────────────
            header_row = [clean_text(c) for c in grid[0]]
            col_map    = map_columns(header_row)

            # ── Chunk generation ─────────────────────────────────────
            chunks = create_chunks(
                grid, col_map,
                region["section"], part_number,
                region["page_no"], table_num
            )
            all_chunks.extend(chunks)
            logger.info(f"Table {table_num} (page {region['page_no']}): {len(chunks)} chunks via Table Transformer.")

        except Exception as e:
            logger.warning(f"Table Transformer failed for table {table_num}: {e}. Using pdfplumber fallback.")
            fb = _pdfplumber_fallback(pdf_path, region, part_number, table_num)
            all_chunks.extend(fb)

    return all_chunks
