import gc
import json
import os
import sys
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Set thread environment variables for stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def parse_pdf(input_pdf, output_json):
    """
    Parses a PDF document using Docling in a single pass.
    
    This replaces the previous page-by-page parsing logic to preserve 
    full table structures and improve extraction accuracy for long documents.
    """
    input_pdf = Path(input_pdf).expanduser().resolve()
    output_json = Path(output_json).expanduser().resolve()

    if not input_pdf.exists():
        print(f"[ERROR] File not found -> {input_pdf}")
        sys.exit(1)

    output_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Parsing entire document in one pass: {input_pdf.name}")

    # Configure Docling pipeline options
    # We now enable do_table_structure since memory is sufficient in Colab.
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,            # Datasheets usually have searchable text
        do_table_structure=True, # Preserve full table structures across pages
        generate_page_images=False,
        generate_picture_images=False,
        num_threads=os.cpu_count() or 4,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
        allowed_formats=[InputFormat.PDF],
    )

    try:
        # Convert the document in a single pass
        result = converter.convert(input_pdf)
        
        # Export Docling document to a dictionary format compatible with the chunker
        document_dict = result.document.export_to_dict()
        
        # Save to output file
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(document_dict, f, indent=2, ensure_ascii=False)
            
        n_texts = len(document_dict.get("texts", []))
        n_tables = len(document_dict.get("tables", []))
        n_pics = len(document_dict.get("pictures", []))
        
        print(f"[INFO] Parsing complete: {n_texts} texts, {n_tables} tables, {n_pics} pictures.")
        print(f"[INFO] Final Docling JSON saved -> {output_json}")
        
        # Optional cleanup
        del result
        gc.collect()

    except Exception as e:
        print(f"[ERROR] Docling failed: {str(e)}")
        # If specific error handling is needed for memory, add it here, 
        # but the goal is to use full pass now.
        raise

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/parse_pdf.py <input_pdf> <output_json>")
        sys.exit(1)

    parse_pdf(sys.argv[1], sys.argv[2])
