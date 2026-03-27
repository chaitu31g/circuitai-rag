"""
hybrid_table_parser_v3.py – Docling + Camelot Hybrid Pipeline
============================================================
Architecture:
  PDF → Docling (section/bbox detection)
      → Camelot-py (precise lattice/stream extraction)
      → Column mapping + cleaning
      → Chunk generation
      → Fallback: pdfplumber-only 
"""

import re
import logging
import camelot
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache

from ingestion.datasheet_chunker import _get_table_contexts, _detect_section, _SKIP_TYPES, Chunk

logger = logging.getLogger(__name__)

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

        # Normalise bbox → [l, t, r, b]
        if isinstance(bbox, dict):
            valid_bbox = [bbox.get("l",0), bbox.get("t",0), bbox.get("r",0), bbox.get("b",0)]
        else:
            valid_bbox = list(bbox[:4])

        # Infer section
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
            "section":              best_sec,
        })

    return regions


# ─────────────────────────────────────────────────────────────────
# 2. TABLE STRUCTURE DETECTION  (Camelot-py)
# ─────────────────────────────────────────────────────────────────
def detect_table_structure_camelot(
    pdf_path: str, 
    page_no: int, 
    bbox: List[float]
) -> List[List[str]]:
    """
    Run Camelot on the specific table region.
    `bbox` is [l, t, r, b] in standard PDF coordinates (72 dpi).
    """
    try:
        # Convert Docling bbox to Camelot string format: "x1,y1,x2,y2"
        # Camelot expects coordinates in standard 72 dpi PDF space.
        # Docling uses standard PDF origin (bottom-left) for its bboxes.
        area = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        
        # Try Lattice mode first (for tables with lines)
        tables = camelot.read_pdf(
            pdf_path,
            pages=str(page_no),
            flavor='lattice',
            table_areas=[area]
        )
        
        # If no tables found with Lattice (lines), try Stream (whitespace)
        if len(tables) == 0:
            tables = camelot.read_pdf(
                pdf_path,
                pages=str(page_no),
                flavor='stream',
                table_areas=[area]
            )
            
        if len(tables) > 0:
            # We take the first table found in the specified area
            return tables[0].df.values.tolist()
            
        return []
    except Exception as e:
        logger.warning(f"Camelot extraction failed for page {page_no}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────
# 3. HEADER DETECTION
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
    "unite":     "unit",
    "value":     "value", # Preserve 'Value' name
    "values":    "value",
    "note":      "condition",
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
    text = re.sub(r'\bC\s*([iors])ss\b', r'C\1ss', text) # C iss -> Ciss
    text = re.sub(r'\bt\s*d\s*\(',      'td(',     text) # t d ( -> td(
    text = re.sub(r'\bR\s*DS\(on\)',     'Rds(on)', text, flags=re.I)
    text = re.sub(r'\bV\s*GS\(th\)',     'Vgs(th)', text, flags=re.I)
    text = re.sub(r'\bV\s*(GS|DS)\b',   r'V\1',    text, flags=re.I)
    text = re.sub(r'\bI\s*D\b',         'ID',      text, flags=re.I)
    text = re.sub(r'(\d+)\s+([A-Za-zΩµ°%]+)\b', r'\1\2', text) # 10 V -> 10V
    
    # Scientific notation 10 0 -> 10^0
    text = re.sub(r'\b10\s+(\d)\b', r'10^\1', text)
    
    # Merged symbols
    text = text.replace("CpF", "pF").replace("V_DSV", "Vds").replace("VDSV", "Vds")
    
    # Remove internal double spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text or "-"
    text = re.sub(r'\bI\s+D\b',      'ID',      text)

    # Scientific notation 10 0 -> 10^0
    text = re.sub(r'\b10\s+(\d)\b', r'10^\1', text)
    
    # Merged units
    text = re.sub(r'\bCpF\b', 'pF', text)
    text = re.sub(r'\bV_DSV\b', 'Vds', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text) # Split e.g. T25 -> T 25
    text = text.replace("V _GS=0 V", "Vgs=0V")
    
    return text.strip() or "-"


def normalize_symbol(symbol: str) -> str:
    if not symbol or symbol == "-": return "-"
    s = symbol.strip().replace(" ", "").replace("Cdsc", "C").replace("IDdsc", "ID")
    return s


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
    cells = [c.strip() for c in row if c]
    if not cells: return False
    
    # Check for "Figure X", "Table X", "Typ. X"
    first = cells[0].lower()
    if re.search(r'\b(figure|fig\.|typ\.|diagram|characteristic)\b', first):
        return True
    
    # Generic section header: a single occupied cell
    non_empty = [c for c in row if c and c != "-"]
    return len(non_empty) == 1 and len(cells) > 3


# ─────────────────────────────────────────────────────────────────
# 8. ROW PROCESSING HELPERS
# ─────────────────────────────────────────────────────────────────
def _resolve_values(row: List[str], col_map: Dict[int, str]):
    """
    Pull min/typ/max/value from named columns.
    """
    named = {name: "-" for name in ["min","typ","max","value"]}
    val_cols = []

    for ci, name in col_map.items():
        if ci >= len(row):
            continue
        v = row[ci].strip() or "-"
        if name in named:
            named[name] = v
        elif name.startswith("val_"):
            val_cols.append(v)

    # Special case: if we have 'value' column, prefer it over positional
    if named["value"] != "-":
        return "-", "-", "-", named["value"]

    # Positional fallback for unlabeled tables
    if all(v == "-" for v in [named["min"], named["typ"], named["max"]]):
        actives = [v for v in val_cols if v != "-"]
        if len(actives) >= 3:
            named["min"], named["typ"], named["max"] = actives[0], actives[1], actives[2]
        elif len(actives) == 2:
            named["typ"], named["max"] = actives[0], actives[1]
        elif len(actives) == 1:
            named["typ"] = actives[0]

    return named["min"], named["typ"], named["max"], "-"


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
    last_parameter = "Unknown"
    last_symbol    = "-"
    last_unit      = "-"
    current_section = section.replace("_", " ").title()

    for row in grid[1:]:          # skip header row
        cells = [c.strip() or "-" for c in row]
        if not cells or all(c == "-" for c in cells):
            continue
            
        # Ignore axis-heavy graph data (too many numbers)
        num_cells = sum(1 for c in cells if any(ch.isdigit() for ch in c))
        if len(cells) > 10 and num_cells > 8:
            continue
            
        if is_section_header_row(cells):
            current_section = cells[0]
            last_condition = "-"
            last_parameter = "Unknown"
            last_symbol = "-"
            last_unit = "-"
            continue
        if len(cells) < 3:
            continue

        # --- Semantic Forward Fill Logic ---
        raw_param = _get_cell(cells, col_map, "parameter")
        if raw_param != "-":
            last_parameter = raw_param
        param = last_parameter

        raw_symbol = normalize_symbol(_get_cell(cells, col_map, "symbol"))
        if raw_symbol != "-":
            last_symbol = raw_symbol
        symbol = last_symbol
        
        # Correction for parameter name drift
        if symbol in ("Ciss", "Coss", "Crss", "C") and "current" in param.lower():
            param = "Capacitance"
        elif symbol in ("ID", "Id") and "capacitance" in param.lower():
            param = "Continuous drain current"

        cond_raw = _get_cell(cells, col_map, "condition")
        if cond_raw != "-":
            condition = clean_text(cond_raw)
            last_condition = condition
        else:
            condition = last_condition

        raw_unit = _get_cell(cells, col_map, "unit")
        if raw_unit != "-":
            last_unit = raw_unit
        unit = last_unit
        
        condition, unit = extract_unit(condition, unit)

        min_val, typ_val, max_val, single_val = _resolve_values(cells, col_map)

        chunk_lines = [
            f"Section: {current_section}",
            f"Parameter: {param}",
            f"Symbol: {symbol}" if symbol != "-" else "",
            f"Condition: {condition}" if condition != "-" else "",
        ]
        if single_val != "-":
            chunk_lines.append(f"Value: {single_val}")
        else:
            chunk_lines.extend([
                f"Min: {min_val}",
                f"Typ: {typ_val}",
                f"Max: {max_val}",
            ])
        if unit == "-" and single_val != "-" and any(u in single_val for u in ("pF","V","A","W")):
            # Extract unit from Value if the Unit column was missed
            match = re.search(r'([A-Za-zΩµ°]+)$', single_val)
            if match:
                unit = match.group(1)
                single_val = single_val[:match.start()].strip()
            
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
                    min_val, typ_val, max_val, single_val = "-", "-", "-", "-"
                    
                    # Logic: if only one value column or header mapping suggests 'Value'
                    if len(row) == 5: # Param, Symbol, Cond, Value, Unit
                        single_val = row[3]
                    elif len(active_vals) >= 3:
                        min_val, typ_val, max_val = active_vals[0], active_vals[1], active_vals[2]
                    elif len(active_vals) == 2:
                        typ_val, max_val = active_vals[0], active_vals[1]
                    elif len(active_vals) == 1:
                        typ_val = active_vals[0]

                    chunk_lines = [
                        f"Section: {current_section}",
                        f"Parameter: {param}",
                        f"Symbol: {symbol}" if symbol != "-" else "",
                        f"Condition: {condition}" if condition != "-" else "",
                    ]
                    if single_val != "-":
                        chunk_lines.append(f"Value: {single_val}")
                    else:
                        chunk_lines.extend([
                            f"Min: {min_val}", f"Typ: {typ_val}", f"Max: {max_val}"
                        ])
                    if unit != "-":
                        chunk_lines.append(f"Unit: {unit}")
                    
                    row_txt = "\n".join(ln for ln in chunk_lines if ln)
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



def extract_tables_hybrid_v3(
    pdf_path:     str,
    docling_data: dict,
    part_number:  str,
) -> List[Chunk]:
    """
    Full pipeline: Docling (detection) → Camelot (extraction) → Chunks.
    Falls back to simple pdfplumber if Camelot yields nothing.
    """
    all_chunks: List[Chunk] = []
    regions = extract_table_regions(docling_data)

    for region in regions:
        table_num = region["table_index"] + 1

        try:
            # ── Camelot extraction within Docling region ─────────────
            grid = detect_table_structure_camelot(
                pdf_path, region["page_no"], region["bbox"]
            )

            if not grid:
                raise ValueError("Camelot returned an empty grid for this region.")

            # ── Header → column map ──────────────────────────────────
            # Clean text in each cell
            clean_grid = [[clean_text(cell) for cell in row] for row in grid]
            header_row = clean_grid[0]
            col_map    = map_columns(header_row)

            # ── Chunk generation ─────────────────────────────────────
            chunks = create_chunks(
                clean_grid, col_map,
                region["section"], part_number,
                region["page_no"], table_num
            )
            all_chunks.extend(chunks)
            logger.info(f"Table {table_num} (page {region['page_no']}): {len(chunks)} chunks via Camelot.")

        except Exception as e:
            logger.warning(f"Camelot failed for table {table_num} on page {region['page_no']}: {e}. Falling back to pdfplumber.")
            fb = _pdfplumber_fallback(pdf_path, region, part_number, table_num)
            all_chunks.extend(fb)

    return all_chunks
