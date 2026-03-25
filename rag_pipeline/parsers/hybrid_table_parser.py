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
            
        data = matched_pt["data"]
        
        # Cleanup rows: convert None to "" and strip whitespace
        clean_rows = []
        for r in data:
            if r:
                clean_row = [str(c).replace('\n', ' ').strip() if c is not None else "" for c in r]
                # Check if row is not completely empty
                if any(clean_row):
                    clean_rows.append(clean_row)
                    
        if len(clean_rows) < 2:
            continue
            
        from rag_pipeline.utils.parameter_extractor import _scrub_and_ffill, _is_subheader_row, _SPAN_TOKENS
        
        # Detect and merge 2-row headers if they exist
        headers = clean_rows[0]
        data_start_idx = 1
        
        if len(clean_rows) > 1 and _is_subheader_row(clean_rows[1], headers):
            sub = clean_rows[1]
            raw_headers = list(headers)
            span_idx = next(
                (i for i, h in enumerate(raw_headers) if any(kw in h.lower() for kw in _SPAN_TOKENS)),
                None
            )
            
            _SUBHDR = {"min", "typ", "max", "unit", "limit", "value", "min.", "typ.", "max."}
            sub_tokens = [s.strip() for s in sub if s.strip() and s.lower().rstrip('.') in _SUBHDR]
            
            if span_idx is not None and sub_tokens:
                before = list(raw_headers[:span_idx])
                after = [p.strip() for p in raw_headers[span_idx + 1:] if p.strip()]
                headers = before + sub_tokens + after
            else:
                length = max(len(raw_headers), len(sub))
                merged = []
                for idx in range(length):
                    p = raw_headers[idx].strip() if idx < len(raw_headers) else ""
                    s = sub[idx].strip() if idx < len(sub) else ""
                    if s and s.lower().rstrip(".") in _SUBHDR:
                        merged.append(s)
                    elif s and not p:
                        merged.append(s)
                    else:
                        merged.append(p)
                headers = merged
            data_start_idx = 2

        rows_to_process = clean_rows[data_start_idx:]
        
        if not rows_to_process:
            continue
            
        # Repair the grid holes via pandas forward fill
        rows_to_process = _scrub_and_ffill(rows_to_process, headers)
        
        # Build Standard table_markdown chunk for the LLM visually
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
        
        # Step 5 & 6: Convert to highly-structured single-row chunks
        safe_headers = [h if h else f"col{idx}" for idx, h in enumerate(headers)]
        
        for row_idx, row in enumerate(rows_to_process):
            if not any(row): 
                continue
                
            chunk_lines = [
                f"Section: {best_sec.replace('_', ' ').title()}",
                f"Table: {table_title}"
            ]
            
            param_val = row[0] if len(row) > 0 else "Unknown"
            symbol_val = row[1] if len(row) > 1 else ""
            unit_val = row[-1] if len(row) > 2 else ""
            
            chunk_lines.append(f"Parameter: {param_val}")
            if symbol_val and symbol_val != "-":
                chunk_lines.append(f"Symbol: {symbol_val}")
                
            conditions = []
            
            # Iterate through the middle items and attach conditions/values
            for col_idx in range(2, len(row) - 1):
                col_name = safe_headers[col_idx]
                val = row[col_idx]
                if not val or val == "-":
                    continue
                    
                if "condition" in col_name.lower() or "test" in col_name.lower():
                    conditions.append(val)
                elif "min" in col_name.lower():
                    chunk_lines.append(f"Minimum: {val} {unit_val}".strip())
                elif "typ" in col_name.lower():
                    chunk_lines.append(f"Typical: {val} {unit_val}".strip())
                elif "max" in col_name.lower():
                    chunk_lines.append(f"Maximum: {val} {unit_val}".strip())
                elif "value" in col_name.lower():
                    chunk_lines.append(f"Value: {val} {unit_val}".strip())
                else:
                    chunk_lines.append(f"{col_name.title()}: {val}")
                    
            if unit_val and unit_val != "-":
                chunk_lines.append(f"Unit: {unit_val}")
            
            if conditions:
                chunk_lines.append(f"Conditions: {'; '.join(conditions)}")

            row_txt = "\n".join(chunk_lines)

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
