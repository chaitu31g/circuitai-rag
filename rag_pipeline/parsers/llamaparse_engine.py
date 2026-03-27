"""
llamaparse_engine.py – Zero-Fail Colab Edition
=============================================
Search Order for API Key:
  1. Google Colab Secrets (userdata.get)
  2. Environment Variables (os.getenv)
  3. Absolute .env path (dotenv)
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
    """Rigorous 3-tier search for the LLAMA_CLOUD_API_KEY."""
    key = None
    
    # Tier 1: Colab Secrets
    if COLAB_AVAILABLE:
        try:
            key = userdata.get('LLAMA_CLOUD_API_KEY')
            if key:
                print("🔑 DEBUG: Found API key in Colab Secrets.")
                return key
        except Exception:
            pass

    # Tier 2: Environment Variable
    key = os.getenv("LLAMA_CLOUD_API_KEY")
    if key and key != "YOUR_KEY_HERE":
        print("🔑 DEBUG: Found API key in Environment Variables.")
        return key
        
    # Tier 3: .env file
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        key = os.getenv("LLAMA_CLOUD_API_KEY")
        if key and key != "llx-4aG7TVeOlsjrXgtf9pFav5hZTNNENyA7MiHzpp136f4gfIQi":
            print(f"🔑 DEBUG: Found API key in .env at {env_path}")
            return key
            
    return ""

# --- CONFIGURATION ---
_LLAMA_INSTRUCTIONS = (
    "Extract all electrical characteristic tables from this semiconductor datasheet. "
    "Columns: Parameter, Symbol, Conditions, Value, Unit."
)

_CLEAN_FIXES = {
    r"\bI\s+D\b": "ID", 
    r"\bt\s+d\(off\)\b": "td(off)",
    r"T\s*=\s*25\s*°C": "T=25°C",
    r"T\s*=\s*70\s*°C": "T=70°C"
}

# ─────────────────────────────────────────────────────────────────
# 1. HARDENED PARSER CORE
# ─────────────────────────────────────────────────────────────────

async def parse_pdf_with_llamaparse(pdf_path: str) -> str:
    api_key = get_api_key()
    
    if not api_key:
        error_msg = (
            "❌ LLAMA_CLOUD_API_KEY is missing! \n"
            "FIX: Go to Colab (Key icon 🔑) and add 'LLAMA_CLOUD_API_KEY' \n"
            "OR create a .env file in the project root."
        )
        raise ValueError(error_msg)

    print(f"🚀 LlamaParse starting for: {os.path.basename(pdf_path)}")
    
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        parsing_instruction=_LLAMA_INSTRUCTIONS,
        max_timeout=5000
    )
    
    documents = await parser.aload_data(pdf_path)
    if not documents:
        raise RuntimeError(f"❌ No data returned from LlamaParse.")
        
    return "\n\n".join([doc.text for doc in documents])

# [Rest of the functional processing logic remains the same...]
# ─────────────────────────────────────────────────────────────────
# 2. MARKDOWN RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────

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

def process_llamaparse_tables(tables: List[List[List[str]]]) -> List[Chunk]:
    all_chunks = []
    for i, tbl in enumerate(tables):
        if len(tbl) < 2: continue
        
        header = tbl[0]
        col_map = map_columns_llamaparse(header)
        is_range = any(k in ["min", "typ", "max"] for k in col_map.keys() if col_map[k] != -1)
        
        l_p, l_s, l_c, l_u = "-", "-", "-", "-"
        
        for row in tbl[1:]:
            if not any(row): continue
            
            def get_cell(key):
                idx = col_map.get(key, -1)
                return clean_cell_llamaparse(row[idx]) if idx != -1 and idx < len(row) else "-"

            p, s, c, u = get_cell("parameter"), get_cell("symbol"), get_cell("condition"), get_cell("unit")
            
            p = p if p != "-" else l_p
            s = s if s != "-" else l_s
            c = c if (c != "-" or p != l_p) else l_c
            u = u if u != "-" else l_unit
            
            if p == "-": continue
            l_p, l_s, l_c, l_u = p, s, c, u

            extracted = resolve_row_values_llamaparse(row, col_map, is_range, u)
            if not extracted: continue
            
            f_unit = extracted.get("unit") or u
            chunk_text = f"Parameter: {p}\nSymbol: {s}\nCondition: {c}\n"
            
            if is_range:
                for k in ["min", "typ", "max"]:
                    if extracted.get(k): chunk_text += f"{k.capitalize()}: {extracted[k]}\n"
            else:
                chunk_text += f"Value: {extracted.get('value', '-')}\n"
            
            if f_unit != "-": chunk_text += f"Unit: {f_unit}"
            
            all_chunks.append(Chunk(
                text=chunk_text.strip(),
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
    
    if mapping["parameter"] == -1: mapping["parameter"] = 0
    if mapping["symbol"] == -1: mapping["symbol"] = 1
    return mapping

def resolve_row_values_llamaparse(row, col_map, is_range, default_unit) -> Dict[str, str]:
    res = {}
    if is_range:
        for k in ["min", "typ", "max"]:
            idx = col_map[k]
            if idx != -1 and idx < len(row) and row[idx].strip() not in ["-", ""]:
                val, unt = split_value_unit_llamaparse(row[idx])
                res[k] = val
                if unt != "-": res["unit"] = unt
    else:
        v_idx = col_map.get("value", col_map.get("typ", -1))
        if v_idx != -1 and v_idx < len(row) and row[v_idx].strip() not in ["-", ""]:
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

def run_llamaparse_extraction(pdf_path: str, part_number: str) -> List[Chunk]:
    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        md_text = loop.run_until_complete(parse_pdf_with_llamaparse(pdf_path))
        tables = extract_tables_from_markdown(md_text)
        return process_llamaparse_tables(tables)
    except Exception as e:
        print(f"💥 LLAMAPARSE CRITICAL FAILURE: {str(e)}")
        raise
