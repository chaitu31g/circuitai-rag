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

# ── LlamaParse Instruction ──────────────────────────────────────────────────
_STRICT_SEMICONDUCTOR_INSTRUCTION = (
    "Extract all semiconductor characteristic and rating tables strictly.\n"
    "Schema Rules:\n"
    "1. Columns MUST be exactly: Parameter | Symbol | Condition | Value | Unit.\n"
    "2. DO NOT include Min, Typ, Max, or combined columns. Pick the most representative single value.\n"
    "3. Repeat labels for every row (un-merge cells).\n"
    "4. DO NOT USE LATEX for symbols. Use plain text (ID, VGS, TA, TJ).\n"
    "5. NO backslashes in symbols. NO underscores like I\\_D. ALWAYS USE plain characters.\n"
    "6. If a symbol is missing, use '-'.\n"
)

async def parse_pdf_with_llamaparse(pdf_path: str) -> str:
    # ── [Existing API Key validation - kept same for brevity] ────────────────
    api_key = get_api_key()
    if not api_key: raise ValueError("❌ LLAMA_CLOUD_API_KEY is missing!")

    print(f"🚀 LlamaParse starting for: {os.path.basename(pdf_path)}")
    
    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        system_prompt_append=_STRICT_SEMICONDUCTOR_INSTRUCTION,
        max_timeout=5000,
        use_vendor_multimodal_model=True,
    )
    
    documents = await parser.aload_data(pdf_path)
    if not documents: raise RuntimeError("❌ No data returned from LlamaParse.")
    
    md_text = "\n\n".join([doc.text for doc in documents])
    
    # ── De-LaTeX Post-Processor ──────────────────────────────────────────────
    # LlamaParse often emits symbols as 'I\_D' or 'V\_GS' which confuses RAG.
    md_text = md_text.replace("\\_", "_").replace("\\ ", " ")
    md_text = re.sub(r"([a-zA-Z])\\([a-zA-Z])", r"\1\2", md_text)
    return md_text

def extract_tables_from_markdown(md_text: str) -> List[List[List[str]]]:
    tables = []
    lines = md_text.split("\n")
    curr_table = []
    in_table = False
    for line in lines:
        line = line.strip()
        if "|" in line:
            # Check for header separator
            if "-|-" in line or "|---" in line or "| :---" in line:
                in_table = True
                continue
            row = [c.strip() for c in line.split("|")]
            # Filter empty boundary elements from markdown split
            if line.startswith("|"): row = row[1:]
            if line.endswith("|"): row = row[:-1]
            
            if not any(row): continue
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
    # Process LlamaParse markdown tables using strict structured extraction and dynamic columns.
    all_chunks = []
    seen_rows = set()
    total_raw_rows = 0
    
    def normalize_symbol(sym: str) -> str:
        if not sym or sym == "-": return "-"
        sym = sym.replace("\\_", "").replace("\\", "").replace("_", "")
        sym = re.sub(r"\s+", "", sym)
        sym = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", sym)
        return sym or "-"

    def normalize_condition(text: str) -> str:
        if not text or text == "-": return "-"
        text = text.replace("\\_", "_").replace("\\", "").replace("Condition:", "")
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("T = ", "T=")
        return text or "-"

    for i, tbl in enumerate(tables):
        if len(tbl) < 2: continue
        
        raw_header = [h.strip() for h in tbl[0]]
        header = [h.lower() for h in raw_header]
        
        p_idx, s_idx = -1, -1
        for idx, h in enumerate(header):
            if "parameter" in h: p_idx = idx
            elif "symbol" in h: s_idx = idx
            
        if p_idx == -1 or s_idx == -1: continue

        l_p, l_s, l_c = "-", "-", "-"
        for row in tbl[1:]:
            total_raw_rows += 1
            if not any(row): continue
            
            p = row[p_idx].strip() if p_idx < len(row) else "-"
            s = row[s_idx].strip() if s_idx < len(row) else "-"
            
            p = p if p != "-" else l_p
            s = s if s != "-" else l_s
            
            p = p.strip()
            sn = normalize_symbol(s)
            
            if p == "-" or p == "" or sn == "-": continue
            l_p, l_s = p, s
            
            row_data = {}
            has_value_column = False
            for col_idx, col_name in enumerate(raw_header):
                if not col_name: continue
                val = row[col_idx].strip() if col_idx < len(row) else "-"
                if val == "-": continue
                
                lower_col = col_name.lower()
                if "condition" in lower_col or "test" in lower_col:
                    val = normalize_condition(val)
                    if val != "-":
                        l_c = val
                    else:
                        val = l_c
                
                if "min" in lower_col or "typ" in lower_col or "max" in lower_col or "value" in lower_col:
                    has_value_column = True
                
                # Dynamic column capture
                row_data[col_name] = val
                
            if not has_value_column: continue

            # Create unique hash
            hash_parts = []
            for k in sorted(row_data.keys()):
                hash_parts.append(f"{k.lower()}:{str(row_data[k]).lower()}")
            row_key_tuple = tuple(hash_parts)
            
            if row_key_tuple in seen_rows: continue
            seen_rows.add(row_key_tuple)
            import json
            ctext = json.dumps(row_data)
            metadata = {
                "component": part_number,
                "type": "table",
                "source": "llamaparse",
                "parameter": p,
                "symbol": sn
            }
            # Inject dynamic columns into metadata
            for k, v in row_data.items():
                metadata[k.lower()] = str(v)
                
            all_chunks.append(Chunk(text=ctext.strip(), chunk_type="parameter_row", metadata=metadata))
            
    print(f"DEBUG: Number of rows before deduplication: {total_raw_rows}")
    print(f"DEBUG: Number of rows after deduplication: {len(all_chunks)}")
    if all_chunks:
        print(f"DEBUG: Sample stored document: {all_chunks[0].text}")
        
    return all_chunks

def run_llamaparse_extraction(pdf_path: str, part_number: str) -> List[Chunk]:
    try:
        try: loop = asyncio.get_event_loop()
        except RuntimeError: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        md = loop.run_until_complete(parse_pdf_with_llamaparse(pdf_path))
        return process_llamaparse_tables(extract_tables_from_markdown(md), part_number)
    except Exception as e: print(f"💥 LLAMAPARSE CRITICAL FAILURE: {str(e)}"); raise
