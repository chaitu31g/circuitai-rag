import json
from pathlib import Path
from rag_pipeline.utils.parameter_extractor import extract_parameter_rows

def test_headers():
    json_path = Path("docling_output/nmos_infineon.json")
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        
    tables = data.get("tables", [])
    for idx, tbl in enumerate(tables):
        cells = tbl.get("data", {}).get("table_cells", [])
        rows_map = {}
        for c in cells:
            r = c.get("row_index", c.get("start_row_offset_idx", 0))
            if r not in rows_map: rows_map[r] = []
            rows_map[r].append(c)
            
        sorted_rows = [
            [c.get("text", "").strip() for c in sorted(rows_map[r], key=lambda x: x.get("col_index", x.get("start_col_offset_idx", 0)))]
            for r in sorted(rows_map)
        ]
        
        if sorted_rows:
            print(f"Table {idx} row 0: {sorted_rows[0]}")
            print(f"Table {idx} row 1: {sorted_rows[1] if len(sorted_rows)>1 else []}")
            print("---")

if __name__ == "__main__":
    test_headers()
