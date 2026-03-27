"""
llamaparse_engine.py – Semiconductor-Specific LlamaParse Engine
==============================================================
Architecture:
  1. LlamaParse → Markdown Tables
  2. Markdown → Structured JSON (List[Dict])
  3. Semantic Washing (ID, Vds/Vgs, T=25C)
  4. Context Inheritance (Forward-fill parameter/symbol)
"""

import os
import re
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from ingestion.datasheet_chunker import Chunk

logger = logging.getLogger(__name__)

# --- PARSING INSTRUCTION ---
_LLAMA_INSTRUCTIONS = (
    "Extract all electrical characteristic tables from this semiconductor datasheet. "
    "Ensure tables have columns: 'Parameter', 'Symbol', 'Conditions' (or 'Test Condition'), "
    "'min', 'typ', 'max', and 'Unit'. "
    "If a parameter has multiple rows of conditions (e.g. TA=25C and TA=70C), "
    "ensure the parameter name is repeated or can be inferred easily from context. "
    "Do not hallucinate data. Preserve exact engineering symbols like ID, Vds, Vgs, and temperature values."
)

_CLEAN_FIXES = {
    r"\bI\s+D\b": "ID", 
    r"\bt\s+d\(off\)\b": "td(off)",
    r"T\s*=\s*25\s*°C": "T=25°C",
    r"T\s*=\s*70\s*°C": "T=70°C",
    r"V\s*DS\s*=\s*(\d+)\s*V": r"Vds=\1V",
    r"V\s*GS\s*=\s*(\d+)\s*V": r"Vgs=\1V"
}

# ─────────────────────────────────────────────────────────────────
# 1. LLAMAPARSE CORE
# ─────────────────────────────────────────────────────────────────

async def parse_pdf_with_llamaparse(pdf_path: str) -> str:
    """Extract full markdown from PDF using LlamaCloud API."""
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        logger.error("LLAMA_CLOUD_API_KEY missing from environment!")
        return ""
    
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        parsing_instruction=_LLAMA_INSTRUCTIONS,
        max_timeout=5000,
        verbose=True
    )
    
    # We use sync wrapper 'load_data' which calls async internally or is part of Reader
    documents = await parser.aload_data(pdf_path)
    return "\n\n".join([doc.text for doc in documents])

# ─────────────────────────────────────────────────────────────────
# 2. MARKDOWN RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────

def extract_tables_from_markdown(md_text: str) -> List[List[List[str]]]:
    """Parse Markdown text to identify and split individual tables."""
    tables = []
    lines = md_text.split("\n")
    curr_table = []
    in_table = False
    
    for line in lines:
        if "|" in line:
            # We don't want the separator line '---|---|---'
            if "-|-" in line or "|---" in line:
                in_table = True
                continue
            
            # Extract row values
            row = [c.strip() for c in line.split("|") if c.strip() or (line.startswith("|") and line.endswith("|"))]
            # Fix for empty cells being filtered out by split logic
            # This is a bit simpler:
            row = [c.strip() for c in line.split("|")][1:-1]
            if not row: continue
            
            curr_table.append(row)
            in_table = True
        else:
            if in_table and curr_table:
                tables.append(curr_table)
                curr_table = []
                in_table = False
                
    if curr_table: tables.append(curr_table)
    return tables

# ─────────────────────────────────────────────────────────────────
# 3. SEMANTIC CLEANING & CHUNKING
# ─────────────────────────────────────────────────────────────────

def process_llamaparse_tables(tables: List[List[List[str]]], part_number: str) -> List[Chunk]:
    all_chunks = []
    for i, tbl in enumerate(tables):
        if len(tbl) < 2: continue
        
        # 1. Map columns (Param, Symbol, Cond, etc.) from the first row (header)
        header = tbl[0]
        col_map = map_columns_llamaparse(header)
        is_range = any(k in ["min", "typ", "max"] for k in col_map.keys() if col_map[k] != -1)
        
        last_p, last_s, last_c, last_u = "-", "-", "-", "-"
        
        # 2. Iterate data rows
        for row in tbl[1:]:
            # Ensure row has enough cells
            if not any(row): continue
            
            # Helper to get value or '-'
            def get_cell(key):
                idx = col_map.get(key, -1)
                return clean_cell_llamaparse(row[idx]) if idx != -1 and idx < len(row) else "-"

            p, s, c, u = get_cell("parameter"), get_cell("symbol"), get_cell("condition"), get_cell("unit")
            
            # --- CONTEXT INHERITANCE ---
            p = p if p != "-" else last_p
            s = s if s != "-" else last_s
            c = c if (c != "-" or p != last_p) else last_c # Only inherit condition if parameter is same
            u = u if u != "-" else last_unit
            
            if p == "-": continue # Skip rows with no parameter context
            
            last_p, last_s, last_c, last_u = p, s, c, u

            # --- VALUE RESOLUTION ---
            extracted = resolve_row_values_llamaparse(row, col_map, is_range, u)
            if not extracted: continue
            
            final_unit = extracted.get("unit") or u
            chunk_lines = [f"Parameter: {p}", f"Symbol: {s}", f"Condition: {c}"]
            
            if is_range:
                for k in ["min", "typ", "max"]:
                    if extracted.get(k): chunk_lines.append(f"{k.capitalize()}: {extracted[k]}")
            else:
                chunk_lines.append(f"Value: {extracted.get('value', '-')}")
            
            if final_unit != "-": chunk_lines.append(f"Unit: {final_unit}")
            
            all_chunks.append(Chunk(
                text="\n".join(chunk_lines),
                chunk_type="parameter_row",
                metadata={"source": "llamaparse", "table_index": i}
            ))
            
    return all_chunks

def map_columns_llamaparse(header: List[str]) -> Dict[str, int]:
    mapping = {k: -1 for k in ["parameter", "symbol", "condition", "min", "typ", "max", "unit", "value"]}
    for i, h in enumerate(header):
        low = h.lower()
        if "param" in low: mapping["parameter"] = i
        elif "sym" in low: mapping["symbol"] = i
        elif "cond" in low or "test" in low: mapping["condition"] = i
        elif "min" in low: mapping["min"] = i
        elif "typ" in low: mapping["typ"] = i
        elif "max" in low: mapping["max"] = i
        elif "unit" in low: mapping["unit"] = i
        elif "val" in low: mapping["value"] = i
    
    # Defaults if header is missing
    if mapping["parameter"] == -1: mapping["parameter"] = 0
    if mapping["symbol"] == -1: mapping["symbol"] = 1
    return mapping

def resolve_row_values_llamaparse(row, col_map, is_range, default_unit) -> Dict[str, str]:
    res = {}
    if is_range:
        for k in ["min", "typ", "max"]:
            idx = col_map[k]
            if idx != -1 and idx < len(row) and row[idx] != "-" and row[idx].strip():
                val, unt = split_value_unit_llamaparse(row[idx])
                res[k] = val
                if unt != "-": res["unit"] = unt
    else:
        v_idx = col_map.get("value", col_map.get("typ", -1))
        if v_idx != -1 and v_idx < len(row) and row[v_idx] != "-" and row[v_idx].strip():
            val, unt = split_value_unit_llamaparse(row[v_idx])
            res["value"] = val
            if unt != "-": res["unit"] = unt
    return res

def split_value_unit_llamaparse(text: str) -> Tuple[str, str]:
    if not text or text == "-": return "-", "-"
    match = re.search(r"^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(.*)$", text.strip())
    if match:
        val, unt = match.groups()
        return val.strip(), unt.strip() or "-"
    return text.strip(), "-"

def clean_cell_llamaparse(text: str) -> str:
    text = text.strip()
    for pattern, repl in _CLEAN_FIXES.items():
        text = re.sub(pattern, repl, text)
    return text if text else "-"

# ─────────────────────────────────────────────────────────────────
# 4. WRAPPER FOR CHUNKER
# ─────────────────────────────────────────────────────────────────

def run_llamaparse_extraction(pdf_path: str, part_number: str) -> List[Chunk]:
    """Top-level entry point for the new LlamaParse pipeline."""
    logger.info(f"Starting LlamaParse extraction for {pdf_path}…")
    
    try:
        # We need an event loop if not running in one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        md_text = loop.run_until_complete(parse_pdf_with_llamaparse(pdf_path))
        if not md_text:
            return []
        
        tables = extract_tables_from_markdown(md_text)
        logger.info(f"Extracted {len(tables)} tables from LlamaParse Markdown.")
        
        return process_llamaparse_tables(tables, part_number)
        
    except Exception as e:
        logger.error(f"LlamaParse extraction failed: {str(e)}", exc_info=True)
        return []

