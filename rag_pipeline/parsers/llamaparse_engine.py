"""
llamaparse_engine.py – Hardened Key Loader
=========================================
Search Order for API Key:
  1. Google Colab Secrets (userdata)
  2. Absolute .env path (dotenv with OVERRIDE)
  3. Environment Variables (os.getenv)
"""

import os
import re
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse

# Try to import Colab specific userdata
try:
    from google.colab import userdata
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False

from ingestion.datasheet_chunker import Chunk

logger = logging.getLogger(__name__)

def get_api_key() -> str:
    """Ultra-Rigorous 3-tier search for the LLAMA_CLOUD_API_KEY."""
    
    # Tier 1: .env File (Force Override)
    # This ensures that even if Python has a 'stale' empty key, we refresh it.
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        key = os.getenv("LLAMA_CLOUD_API_KEY")
        if key and key.startswith("llx-") and "YOUR_KEY" not in key:
            print(f"🔑 DEBUG: Refreshed key from .env at {env_path}")
            return key

    # Tier 2: Colab Secrets
    if COLAB_AVAILABLE:
        try:
            key = userdata.get('LLAMA_CLOUD_API_KEY')
            if key and key.startswith("llx-"):
                print("🔑 DEBUG: Found key in Colab Secrets.")
                return key
        except Exception:
            pass

    # Tier 3: Direct Environment Variable
    key = os.getenv("LLAMA_CLOUD_API_KEY")
    if key and key.startswith("llx-") and "YOUR_KEY" not in key:
        return key
            
    return ""

# ─────────────────────────────────────────────────────────────────
# [Rest of the extraction logic - same as before]
# ─────────────────────────────────────────────────────────────────

async def parse_pdf_with_llamaparse(pdf_path: str) -> str:
    api_key = get_api_key()
    
    if not api_key:
        error_msg = (
            "❌ LLAMA_CLOUD_API_KEY is missing! \n"
            "FIX: Ensure your .env file in the root has 'LLAMA_CLOUD_API_KEY=llx-...' \n"
            "AND Restart the backend to apply the changes."
        )
        raise ValueError(error_msg)

    print(f"🚀 LlamaParse starting for: {os.path.basename(pdf_path)}")
    
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        parsing_instruction="Extract all semiconductor characteristic tables. Columns: Parameter, Symbol, Conditions, Value, Unit.",
        max_timeout=5000
    )
    
    documents = await parser.aload_data(pdf_path)
    if not documents:
        raise RuntimeError("❌ No data returned from LlamaParse.")
        
    return "\n\n".join([doc.text for doc in documents])

def extract_tables_from_markdown(md_text: str) -> List[List[List[str]]]:
    tables = []
    lines = md_text.split("\n")
    curr_table = []
    in_table = False
    for line in lines:
        if "|" in line:
            if "-|-" in line or "|---" in line:
                in_table = True
                continue
            row = [c.strip() for c in line.split("|")][1:-1]
            if not row or all(not c for c in row): continue
            curr_table.append(row)
            in_table = True
        else:
            if in_table and curr_table:
                tables.append(curr_table)
                curr_table = []
                in_table = False
    if curr_table: tables.append(curr_table)
    return tables

def process_llamaparse_tables(tables: List[List[List[str]]], part_number: str) -> List[Chunk]:
    """Process LlamaParse markdown tables into validated, deduplicated chunks."""
    all_chunks = []
    seen_rows = set() # (parameter, symbol, condition, value)
    
    # ── Noise removal regex ──────────────────────────────────────────────────
    # Removes engineering noise like "GS", "j" and normalized multiple spaces
    def clean_engineering_text(text: str) -> str:
        if not text or text == "-": return "-"
        # Remove common noise patterns seen in LlamaParse datasheet output (e.g., "GS j")
        text = re.sub(r"\bGS\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bj\b", "", text, flags=re.IGNORECASE)
        # Normalize spacing
        text = re.sub(r"\s+", " ", text).strip()
        return text or "-"

    _CLEAN_FIXES = {
        r"\bI\s+D\b": "ID", 
        r"\bt\s+d\(off\)\b": "td(off)", 
        r"T\s*=\s*25\s*°C": "T=25°C", 
        r"T\s*=\s*70\s*°C": "T=70°C"
    }
    
    for i, tbl in enumerate(tables):
        if len(tbl) < 2: continue # Needs header + at least 1 row
        
        header = tbl[0]
        col_map = map_columns_llamaparse(header)
        
        # ── Strict Table Structure Enforcement ────────────────────────────────
        # Must have: Parameter, Symbol, Conditions, and a value column
        # Skip if any core column is missing.
        required_cols = ["parameter", "symbol", "condition"]
        has_required = all(col_map[k] != -1 for k in required_cols)
        has_value = any(col_map[k] != -1 for k in ["value", "min", "typ", "max"])
        
        if not (has_required and has_value):
            logger.warning(f"Skipping Table {i} — missing required columns (Parameter, Symbol, Conditions, or Value).")
            continue

        is_range = any(k in ["min", "typ", "max"] for k in col_map.keys() if col_map[k] != -1)
        l_p, l_s, l_c, l_u = "-", "-", "-", "-"
        
        for row in tbl[1:]:
            if not any(row): continue
            
            def get_cell(key):
                idx = col_map.get(key, -1)
                txt = row[idx].strip() if idx != -1 and idx < len(row) else "-"
                for p, r in _CLEAN_FIXES.items(): 
                    txt = re.sub(p, r, txt)
                return txt
            
            # Extract raw values
            raw_p, raw_s, raw_c, raw_u = get_cell("parameter"), get_cell("symbol"), get_cell("condition"), get_cell("unit")
            
            # Forward-fill logic
            p = raw_p if raw_p != "-" else l_p
            s = raw_s if raw_s != "-" else l_s
            c = raw_c if (raw_c != "-" or raw_p != l_p) else l_c
            u = raw_u if raw_u != "-" else l_u
            
            # ── Clean Condition Text ──────────────────────────────────────────
            c = clean_engineering_text(c)
            
            if p == "-" or p == "": continue
            l_p, l_s, l_c, l_u = p, s, c, u
            
            extr = resolve_row_values_llamaparse(row, col_map, is_range, u)
            if not extr: continue
            
            # Determine primary value for validation and deduplication
            val_str = str(extr.get("value", extr.get("typ", extr.get("max", extr.get("min", "-")))))
            
            # ── Validation: Skip rows where core info is missing or empty ──────
            if p == "-" or s == "-" or val_str == "-" or val_str == "":
                continue
                
            # ── Deduplication ────────────────────────────────────────────────
            # Use unique key: (parameter, symbol, condition, value)
            row_key = (p.lower(), s.lower(), c.lower(), val_str.lower())
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            
            f_u = extr.get("unit") or u
            ctext = f"Parameter: {p}\nSymbol: {s}\nCondition: {c}\n"
            if is_range:
                for k in ["min", "typ", "max"]: 
                    if extr.get(k): ctext += f"{k.capitalize()}: {extr[k]}\n"
            else: 
                ctext += f"Value: {val_str}\n"
            
            if f_u != "-": ctext += f"Unit: {f_u}"
            
            # ── Metadata ────────────────────────────────────────────────────
            # Ensure component is never missing and matches PDF name
            metadata = {
                "component":     part_number,
                "part_number":   part_number,
                "type":          "table",
                "chunk_type":    "parameter_row",
                "source":        "llamaparse",
                "table_index":   i,
                "parameter":     p,
                "symbol":        s
            }
            
            all_chunks.append(Chunk(text=ctext.strip(), chunk_type="parameter_row", metadata=metadata))
            
    return all_chunks

def map_columns_llamaparse(header: List[str]) -> Dict[str, int]:
    m = {k: -1 for k in ["parameter", "symbol", "condition", "min", "typ", "max", "unit", "value"]}
    for i, h in enumerate(header):
        l = h.lower()
        if "param" in l: m["parameter"] = i
        elif "sym" in l: m["symbol"] = i
        elif "cond" in l or "test" in l: m["condition"] = i
        elif "min" in l: m["min"] = i
        elif "typ" in l: m["typ"] = i
        elif "max" in l: m["max"] = i
        elif "unit" in l: m["unit"] = i
        elif "val" in l: m["value"] = i
    if m["parameter"] == -1: m["parameter"] = 0
    if m["symbol"] == -1: m["symbol"] = 1
    return m

def resolve_row_values_llamaparse(row, col_map, is_range, def_u) -> Dict[str, str]:
    res = {}
    def split(t):
        if not t or t == "-": return "-", "-"
        ma = re.search(r"^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(.*)$", t.strip())
        return (ma.groups()[0].strip(), ma.groups()[1].strip() or "-") if ma else (t.strip(), "-")
        
    if is_range:
        for k in ["min", "typ", "max"]:
            idx = col_map[k]
            if idx != -1 and idx < len(row) and row[idx].strip() not in ["-", ""]:
                v, u = split(row[idx]); res[k] = v
                if u != "-": res["unit"] = u
    else:
        vi = col_map.get("value", col_map.get("typ", -1))
        if vi != -1 and vi < len(row) and row[vi].strip() not in ["-", ""]:
            v, u = split(row[vi]); res["value"] = v
            if u != "-": res["unit"] = u
    return res

def run_llamaparse_extraction(pdf_path: str, part_number: str) -> List[Chunk]:
    try:
        try: loop = asyncio.get_event_loop()
        except RuntimeError: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        md = loop.run_until_complete(parse_pdf_with_llamaparse(pdf_path))
        return process_llamaparse_tables(extract_tables_from_markdown(md), part_number)
    except Exception as e: print(f"💥 LLAMAPARSE CRITICAL FAILURE: {str(e)}"); raise
