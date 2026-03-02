import gc
import json
import os
import sys
from pathlib import Path

import fitz
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


BATCH_SIZE = 1

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def split_pdf(input_pdf, batch_size=BATCH_SIZE):
    input_pdf = Path(input_pdf).expanduser().resolve()
    batches = []

    with fitz.open(input_pdf) as doc:
        total_pages = doc.page_count
        for start in range(0, total_pages, batch_size):
            end = min(start + batch_size - 1, total_pages - 1)
            temp_pdf = input_pdf.parent / f"__temp_{start + 1}_{end + 1}.pdf"

            mini = fitz.open()
            mini.insert_pdf(doc, from_page=start, to_page=end)
            mini.save(temp_pdf)
            mini.close()

            batches.append(temp_pdf)

    return batches


def extract_tables_pymupdf(input_pdf):
    """Extract tables directly from the PDF using PyMuPDF's find_tables().

    This bypasses Docling's ONNX-based table structure model, which crashes
    with std::bad_alloc on low-memory systems.

    Returns a list of table dicts compatible with the Docling table format
    used by the downstream ingestion pipeline.
    """
    input_pdf = Path(input_pdf).expanduser().resolve()
    all_tables = []

    with fitz.open(input_pdf) as doc:
        for page_idx, page in enumerate(doc):
            tabs = page.find_tables()
            for t_idx, table in enumerate(tabs.tables):
                raw_rows = table.extract()
                if len(raw_rows) < 2:
                    continue

                # Build table_cells in Docling-compatible format.
                cells = []
                for r_idx, row in enumerate(raw_rows):
                    for c_idx, cell_text in enumerate(row):
                        cells.append({
                            "text": (cell_text or "").replace("\n", " ").strip(),
                            "start_row_offset_idx": r_idx,
                            "start_col_offset_idx": c_idx,
                        })

                all_tables.append({
                    "data": {"table_cells": cells},
                    "prov": [{"page_no": page_idx + 1}],
                })

    return all_tables


def _extract_text_pymupdf(pdf_path, page_offset=0):
    """Fallback: extract text blocks from a single-page PDF using PyMuPDF.

    Returns a Docling-compatible dict with texts populated from raw text blocks.
    """
    texts = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            blocks = page.get_text("blocks")
            for idx, block in enumerate(blocks):
                text = block[4].strip() if len(block) > 4 else ""
                if not text:
                    continue
                texts.append({
                    "text": text,
                    "label": "text",
                    "self_ref": f"#/texts/{idx}",
                    "prov": [{"page_no": page_offset + 1}],
                })

    return {"texts": texts, "tables": [], "pictures": []}


def merge_docling(temp_json_paths):
    if not temp_json_paths:
        return {}

    # Load the first batch as the base
    with open(temp_json_paths[0], "r", encoding="utf-8") as f:
        merged = json.load(f)

    for key in ("texts", "tables", "pictures"):
        merged.setdefault(key, [])

    # Iteratively load, append, and discard subsequent batches
    for file_path in temp_json_paths[1:]:
        with open(file_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        merged["texts"].extend(result.get("texts", []))
        merged["tables"].extend(result.get("tables", []))
        merged["pictures"].extend(result.get("pictures", []))

    return merged


def parse_pdf(input_pdf, output_json):
    input_pdf = Path(input_pdf).expanduser().resolve()
    output_json = Path(output_json).expanduser().resolve()

    if not input_pdf.exists():
        print(f"[ERROR] File not found -> {input_pdf}")
        sys.exit(1)

    output_json.parent.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Extract tables with PyMuPDF (no AI models) ---
    print("[INFO] Extracting tables with PyMuPDF...")
    pymupdf_tables = extract_tables_pymupdf(input_pdf)
    print(f"   found {len(pymupdf_tables)} tables")

    # --- Step 2: Parse text/pictures with Docling (no OCR, no table model) ---
    print(f"[INFO] Splitting PDF into {BATCH_SIZE}-page batches...")
    batches = split_pdf(input_pdf, BATCH_SIZE)

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=False,
        generate_page_images=False,
        generate_picture_images=False,
        num_threads=1,
        images_scale=1.0,  # Reduce resolution to save memory (default is 2.0)
    )
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
        allowed_formats=[InputFormat.PDF],
    )
    temp_json_paths = []

    try:
        for i, batch in enumerate(batches):
            print(f"[INFO] Parsing batch -> {batch.name}")
            try:
                result = converter.convert(batch)

                # Save dict immediately to disk to free memory
                temp_json = input_pdf.parent / f"__temp_dict_{i}.json"
                with temp_json.open("w", encoding="utf-8") as f:
                    json.dump(result.document.export_to_dict(), f)
                temp_json_paths.append(temp_json)

                # Delete references and force garbage collection
                del result
            except Exception as e:
                print(f"[WARN] Docling failed on {batch.name}: {e}")
                print(f"[INFO] Using PyMuPDF text fallback for page {i + 1}")
                # Fall back to PyMuPDF plain text extraction
                fallback = _extract_text_pymupdf(batch, page_offset=i)
                temp_json = input_pdf.parent / f"__temp_dict_{i}.json"
                with temp_json.open("w", encoding="utf-8") as f:
                    json.dump(fallback, f)
                temp_json_paths.append(temp_json)
            finally:
                gc.collect()

        print("[INFO] Merging batches...")
        merged = merge_docling(temp_json_paths)

        # --- Step 3: Replace empty Docling tables with PyMuPDF tables ---
        merged["tables"] = pymupdf_tables
        print(f"[INFO] Injected {len(pymupdf_tables)} PyMuPDF tables into output")

        with output_json.open("w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)

        print(f"[INFO] Final Docling JSON saved -> {output_json}")
    finally:
        for batch in batches:
            try:
                batch.unlink(missing_ok=True)
            except Exception:
                pass
        for temp_json in temp_json_paths:
            try:
                temp_json.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/parse_pdf.py <input_pdf> <output_json>")
        sys.exit(1)

    parse_pdf(sys.argv[1], sys.argv[2])
