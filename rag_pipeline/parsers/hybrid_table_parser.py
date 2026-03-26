import logging
import re
from typing import List, Dict, Any

from rag_pipeline.parsers.pdftable import PdfTable
from ingestion.datasheet_chunker import _get_table_contexts, _detect_section, _SKIP_TYPES, Chunk

logger = logging.getLogger(__name__)

def match_docling_table_with_pdftable(
    doc_table: dict, 
    pdf_tables: List[Dict[str, Any]], 
    docling_page_rank: int
) -> Dict[str, Any]:
    """
    Match Docling tables with PdfTable tables sequentially by page.
    Bypasses coordinate mismatch problems.
    docling_page_rank: This is the Nth table Docling found on this page (0-indexed).
    """
    prov = (doc_table.get("prov") or [{}])[0]
    page_no = prov.get("page_no")
    
    if not page_no:
        return None

    # Get all pdfplumber tables on this exact page
    page_pdf_tables = [pt for pt in pdf_tables if pt["page"] == page_no]
    
    if page_pdf_tables:
        # Match sequentially: 1st docling table = 1st pdfplumber table
        rank = min(docling_page_rank, len(page_pdf_tables) - 1)
        return page_pdf_tables[rank]

    # Fallback: if no tables found on exact page, match nearest by absolute page distance
    if not pdf_tables:
        return None
        
    return min(pdf_tables, key=lambda t: abs(t["page"] - page_no))

def is_section_header_row(row):
    """
    Detect non-data rows such as section headers (e.g., 'Dynamic characteristics').
    Usually, it's a row where only the first column has text, and no numbers.
    """
    if not row:
        return False
    cells = [str(cell).strip() for cell in row if cell is not None]
    if len(cells) == 0:
        return False
        
    has_text = len(cells[0]) > 0
    others_empty = all(len(c) == 0 or c == "-" for c in cells[1:])
    has_number = any(any(char.isdigit() for char in cell) for cell in cells)
    
    # If there are no numbers and only the first cell has text, it's a section header
    if has_text and others_empty and not has_number:
        return True
        
    return False

def unmerge_multiline_row(row):
    """
    Expands a single row with multi-line cells (e.g., "0.23\\n0.18") into multiple rows.
    """
    cells = [str(c).strip() if c is not None else "" for c in row]
    lines_per_cell = [c.split('\n') for c in cells]
    max_lines = max(len(lines) for lines in lines_per_cell)
    
    if max_lines <= 1:
        return [[c.replace('\n', ' ').strip() for c in cells]]
        
    expanded = []
    for i in range(max_lines):
        new_row = []
        for lines in lines_per_cell:
            if i < len(lines):
                new_row.append(lines[i].strip())
            else:
                new_row.append("")
        expanded.append(new_row)
    return expanded

def clean_condition(text):
    if not text or text == "-":
        return "-"
    text = text.replace("V =60 V, DS V =0 V", "Vds=60V, Vgs=0V")
    text = re.sub(r'V\s*=\s*([-\d.]+)\s*V,\s*DS', r'Vds=\1V', text)
    text = re.sub(r'V\s*=\s*([-\d.]+)\s*V,\s*GS', r'Vgs=\1V', text)
    text = re.sub(r'\s*=\s*', '=', text)
    text = re.sub(r'(?<=\d)\s+([A-Za-z]+)', r'\1', text)
    # clean weird characters like j or nm at end
    text = re.sub(r'\s+[A-Za-z]$', '', text)
    return text.strip()

def normalize_symbol(symbol):
    if not symbol or symbol == "-":
        return "-"
    return symbol.replace(" ", "")

def extract_unit(condition, current_unit):
    unit_pattern = r'\s+(pF|nF|uF|F|ns|us|ms|s|MHz|kHz|Hz|mV|V|mA|uA|A|mW|W|mOhm|Ohm|Ω|mΩ|°C)$'
    match = re.search(unit_pattern, condition, flags=re.IGNORECASE)
    if match:
        extracted = match.group(1)
        condition = condition[:match.start()].strip()
        if current_unit == "-" or not current_unit:
            current_unit = extracted
    return condition, current_unit


def extract_tables_hybrid(
    pdf_path: str,
    docling_data: dict,
    part_number: str
) -> List[Chunk]:
    """
    Hybrid Pipeline:
    1. Parse Docling document structure (passed via docling_data)
    2. Extract accurate tables with PdfTable
    3. Match Docling table to PdfTable output sequentially
    4. Format to structured parameter rows with Context
    5. Return chunks
    """
    logger.info(f"Extracting tables with hybrid PdfTable parser for {part_number}")
    all_chunks: List[Chunk] = []
    
    try:
        pdf_tables = PdfTable().extract(pdf_path)
    except Exception as e:
        logger.error(f"Hybrid table extraction failed: {e}. Falling back to standard docling tables.")
        raise

    tables = docling_data.get("tables", [])
    texts = docling_data.get("texts", [])
    
    # --- Mandatory Diagnostic Logging ---
    logger.info(f"Docling tables: {len(tables)}")
    logger.info(f"PdfTable tables: {len(pdf_tables)}")
    for t in pdf_tables:
        logger.debug(f"PdfTable table -> page: {t['page']}, rows: {len(t['data'])}")
    
    table_contexts = _get_table_contexts(docling_data)
    
    # Track the sequential rank of Docling tables per page
    page_rank_counter = {}

    for i, tbl in enumerate(tables):
        prov = (tbl.get("prov") or [{}])[0]
        page = prov.get("page_no")
        
        # Compute docling page rank
        if page:
            docling_page_rank = page_rank_counter.get(page, 0)
            page_rank_counter[page] = docling_page_rank + 1
        else:
            docling_page_rank = 0

        # Step 4: Extract Table Title & Section (Docling structure)
        ctx = table_contexts.get(i)
        if ctx:
            best_sec, table_title = ctx
        else:
            best_sec = "electrical_characteristics"
            for t in texts:
                t_page = (t.get("prov") or [{}])[0].get("page_no") if t.get("prov") else None
                if t_page == page:
                    d = _detect_section(t.get("text", ""))
                    if d and d not in _SKIP_TYPES:
                        best_sec = d
                        break
            table_title = ""
            
        # Step 3: Match Docling table with PdfTable using page sequentially
        matched_pt = match_docling_table_with_pdftable(tbl, pdf_tables, docling_page_rank)
        
        if not matched_pt:
            logger.warning(f"No PdfTable matched Docling table {i} on page {page}. Falling back to standard Docling format.")
            # Fallback to standard Docling extractor if pdfplumber missed it
            from rag_pipeline.utils.parameter_extractor import extract_parameter_rows
            fallback_chunks = extract_parameter_rows(tbl, best_sec, part_number, i + 1, table_title=table_title)
            all_chunks.extend(fallback_chunks)
            continue
            
        try:
            data = matched_pt["data"]
            
            if not data or len(data) < 2:
                continue
                
            headers = [str(c).replace('\n', ' ').strip() if c is not None else "" for c in data[0]]
            
            clean_rows = []
            for row in data[1:]:
                # Expand multiline cells
                expanded = unmerge_multiline_row(row)
                clean_rows.extend(expanded)
                    
            if not clean_rows:
                continue
                
            from rag_pipeline.utils.parameter_extractor import _scrub_and_ffill
            
            rows_to_process = clean_rows
            
            # Repair the grid holes via pandas forward fill
            # BUT we strictly avoid forward-filling the condition indefinitely 
            # if we can handle it at chunking time. _scrub_and_ffill still handles param/symbol merges.
            rows_to_process = _scrub_and_ffill(rows_to_process, headers)
            
        except Exception as e:
            logger.warning(f"Failed to process matched PdfTable: {e}. Falling back to docling.")
            from rag_pipeline.utils.parameter_extractor import extract_parameter_rows
            fallback_chunks = extract_parameter_rows(tbl, best_sec, part_number, i + 1, table_title=table_title)
            all_chunks.extend(fallback_chunks)
            continue
        
        # Build Standard table_markdown chunk for visual LLM grounding
        from rag_pipeline.utils.parameter_extractor import _rows_to_markdown
        markdown_str = _rows_to_markdown(headers, rows_to_process)
        table_preamble = (
            f"Table {i+1}: {best_sec.replace('_', ' ').title()} data for {part_number}.\n"
            f"This is a strictly generated markdown representation.\n"
        )
        md_text = table_preamble + f"<DATASHEET_TABLE>\n{markdown_str}\n</DATASHEET_TABLE>"
        
        all_chunks.append(Chunk(
            text=md_text,
            chunk_type="table",
            metadata={
                "type": "table_markdown",
                "chunk_type": "table_markdown",
                "part_number": part_number,
                "section": best_sec,
                "table_name": table_title,
                "table_number": i + 1,
                "page": matched_pt["page"]
            }
        ))
        
        # Create highly structured parameter rows
        last_condition = "-"
        current_section = best_sec.replace('_', ' ').title()
        
        for row_idx, r in enumerate(rows_to_process):
            row = [str(x).strip() if x is not None and str(x).strip() != "" else "-" for x in r]
            
            if not row or all(x == "-" for x in row):
                continue
                
            if is_section_header_row(row):
                current_section = row[0]
                last_condition = "-"  # Boundary reset
                continue
                
            if len(row) < 4:
                continue
                
            param_val = row[0] if row[0] != "-" else "Unknown"
            symbol_val = normalize_symbol(row[1])
            
            condition = row[2]
            if condition == "-":
                condition = last_condition
            else:
                condition = clean_condition(condition)
                last_condition = condition
                
            unit_val = row[-1]
            condition, unit_val = extract_unit(condition, unit_val)
            
            # Extract active non-empty value columns
            val_columns = row[3:-1]
            active_vals = [v for v in val_columns if v != "-"]
            
            min_val, typ_val, max_val = "-", "-", "-"
            if len(active_vals) >= 3:
                min_val, typ_val, max_val = active_vals[0], active_vals[1], active_vals[2]
            elif len(active_vals) == 2:
                typ_val, max_val = active_vals[0], active_vals[1]
            elif len(active_vals) == 1:
                typ_val = active_vals[0]
                
            chunk_lines = [
                f"Section: {current_section}",
                f"Table: {table_title}" if table_title else "",
                f"Parameter: {param_val}",
                f"Symbol: {symbol_val}" if symbol_val != "-" else "",
                f"Condition: {condition}" if condition != "-" else "",
                f"Min: {min_val}",
                f"Typ: {typ_val}",
                f"Max: {max_val}",
                f"Unit: {unit_val}" if unit_val != "-" else ""
            ]
            
            row_txt = "\n".join(ln for ln in chunk_lines if ln)
            
            all_chunks.append(Chunk(
                text=row_txt,
                chunk_type="parameter_row",
                metadata={
                    "type": "table_row",
                    "chunk_type": "parameter_row", 
                    "part_number": part_number,
                    "section": best_sec,
                    "table_name": table_title,
                    "table_number": i + 1,
                    "page": matched_pt["page"],
                    "parameter": param_val
                }
            ))

    return all_chunks
