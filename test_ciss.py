import json
from pathlib import Path
from rag_pipeline.utils.parameter_extractor import extract_parameter_rows

def test_param():
    json_path = Path("docling_output/nmos_infineon.json")
    with open(json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
        
    tables = data.get("tables", [])
    for idx, tbl in enumerate(tables):
        rows = extract_parameter_rows(tbl, section_name="Dynamic characteristics", part_number="BSS138N", table_number=idx)
        for r in rows:
            if "Ciss" in r.text or "capacitance" in r.text.lower():
                print(r.text)
                print("-------")

if __name__ == "__main__":
    test_param()
