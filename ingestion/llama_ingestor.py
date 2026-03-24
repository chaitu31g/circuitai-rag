"""
ingestion/llama_ingestor.py
────────────────────────────────────────────────────────────────────────────
Hybrid Pandas Forward-Fill ingestion for semiconductor datasheets.

THE PROBLEM — MERGED CELLS IN DATASHEETS
=========================================
Infineon / ON Semiconductor "Static Characteristics" tables use visually
merged cells. For example, "Drain-source on-state resistance" spans 3 rows
for different V_GS conditions. Standard PDF parsers write the parameter name
in Row 1 but leave the parameter column *empty* in Rows 2 and 3. When the
vector store chunks this text the LLM loses the context of what those rows
belong to.

THE SOLUTION — HYBRID PANDAS FORWARD-FILL
==========================================
1. Docling converts the PDF to a structured document object.
2. Every table is intercepted and converted to a Pandas DataFrame.
3. Empty / whitespace / None cells are replaced with pd.NA.
4. A forward-fill (ffill) is applied **only to the first two columns**
   (Parameter + Symbol) so the parameter name is copied down to the blank
   rows without touching Min/Max/Test-Condition columns.
5. The repaired DataFrame is converted back to a Markdown table and wrapped
   in explicit boundary markers.
6. The general document text and all enhanced tables are stitched together
   and returned as a LlamaIndex Document object.

Usage
-----
    from ingestion.llama_ingestor import ingest_datasheet_to_llama_document

    doc = ingest_datasheet_to_llama_document("path/to/irf540n.pdf")
    # doc is a llama_index.core.schema.Document ready for embedding
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Thread-count guard (prevents Docling OMP contention on multi-core hosts) ─
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_docling_converter():
    """Construct a Docling DocumentConverter with table-structure enabled."""
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions(
        do_ocr=False,               # datasheets have searchable text
        do_table_structure=True,    # ← essential: full table structure / merges
        generate_page_images=False,
        generate_picture_images=False,
        num_threads=os.cpu_count() or 4,
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
        allowed_formats=[InputFormat.PDF],
    )


def _clean_dataframe(df):
    """Replace empty/whitespace/None cells with pd.NA."""
    import pandas as pd

    def _to_na(val):
        if val is None:
            return pd.NA
        s = str(val).strip()
        return pd.NA if s == "" else s

    return df.applymap(_to_na)


def _forward_fill_param_cols(df):
    """
    Forward-fill *only* the first two columns (Parameter, Symbol).

    Rationale
    ---------
    Column 0 = Parameter name (e.g. "Drain-source on-state resistance")
    Column 1 = Symbol        (e.g. "R_DS(on)")

    These are the merge-victim columns. The numeric value/condition columns
    are deliberately excluded to avoid propagating Min/Max values downward.

    Edge cases handled
    ------------------
    * 0 columns  → no-op, return as-is.
    * 1 column   → fill only col[0].
    * 2+ columns → fill col[0] and col[1] only.
    """
    if df.shape[1] == 0:
        return df

    cols_to_fill = df.columns[:min(2, df.shape[1])].tolist()
    df[cols_to_fill] = df[cols_to_fill].ffill(axis=0)
    return df


def _table_to_enhanced_markdown(table_obj) -> Optional[str]:
    """
    Convert a single Docling table to a forward-filled Markdown string.

    Returns None if the table is empty or conversion fails (caller skips it).
    """
    import pandas as pd

    try:
        # ── Step 1: Docling → Pandas ─────────────────────────────────────────
        df: pd.DataFrame = table_obj.export_to_dataframe()

        if df.empty:
            logger.debug("Skipping empty table.")
            return None

        # ── Step 2: Clean ────────────────────────────────────────────────────
        df = _clean_dataframe(df)

        # ── Step 3: Forward-fill first two columns only ──────────────────────
        df = _forward_fill_param_cols(df)

        # ── Step 4: DataFrame → Markdown ─────────────────────────────────────
        # tabulate is required by pandas .to_markdown(); it ships with LlamaIndex.
        try:
            md_table = df.to_markdown(index=False)
        except ImportError:
            # Fallback: plain pipe-separated text if tabulate is absent.
            lines = ["| " + " | ".join(str(c) for c in df.columns) + " |"]
            lines.append("|" + "|".join(["---"] * len(df.columns)) + "|")
            for _, row in df.iterrows():
                lines.append("| " + " | ".join(str(v) if not pd.isna(v) else "" for v in row) + " |")
            md_table = "\n".join(lines)

        # ── Step 5: Tag with boundary markers ────────────────────────────────
        return (
            "\n\n--- ENHANCED DATASHEET TABLE ---\n"
            f"{md_table}\n"
            "--------------------------------\n\n"
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Table conversion failed — skipping this table. Reason: %s",
            exc,
            exc_info=True,
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def ingest_datasheet_to_llama_document(pdf_path: str):
    """
    Parse a semiconductor datasheet PDF and return a LlamaIndex Document.

    Implements the "Hybrid Pandas Forward-Fill" strategy to un-merge visually
    merged table cells before the text is embedded into the vector store.

    Parameters
    ----------
    pdf_path : str
        Absolute or relative path to the PDF file.

    Returns
    -------
    llama_index.core.schema.Document
        A single Document whose `.text` is:
          • The full markdown export of the document's prose/headings, PLUS
          • Each detected table converted to a forward-filled Markdown grid
            and wrapped in ``--- ENHANCED DATASHEET TABLE ---`` markers.
        The `.metadata` dict contains ``pdf_path``, ``filename``,
        ``num_tables``, and ``num_tables_enhanced``.

    Raises
    ------
    FileNotFoundError
        If the PDF does not exist at ``pdf_path``.
    RuntimeError
        If required packages (docling, llama-index-core, pandas) are missing.
    """
    # ── Dependency guard ──────────────────────────────────────────────────────
    try:
        import pandas as _pd  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required for forward-fill ingestion.\n"
            "Install it with:  pip install pandas tabulate"
        ) from exc

    try:
        from llama_index.core.schema import Document
    except ImportError as exc:
        raise RuntimeError(
            "llama-index-core is required.\n"
            "Install it with:  pip install llama-index-core"
        ) from exc

    pdf_path_obj = Path(pdf_path).expanduser().resolve()
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path_obj}")

    logger.info("ingest_datasheet_to_llama_document: parsing %s", pdf_path_obj.name)

    # ── Step 1: Docling parsing ───────────────────────────────────────────────
    converter = _build_docling_converter()
    result = converter.convert(pdf_path_obj)
    doc_obj = result.document

    # ── Step 2: Extract general prose as Markdown ─────────────────────────────
    general_text: str = doc_obj.export_to_markdown()

    # ── Step 3 & beyond: Intercept every table ───────────────────────────────
    tables = doc_obj.tables          # list of Docling TableItem objects
    num_tables = len(tables)
    enhanced_blocks: list[str] = []

    for idx, table in enumerate(tables):
        logger.debug("Processing table %d / %d …", idx + 1, num_tables)
        block = _table_to_enhanced_markdown(table)
        if block is not None:
            enhanced_blocks.append(block)

    num_enhanced = len(enhanced_blocks)
    logger.info(
        "%s: %d table(s) found, %d enhanced successfully.",
        pdf_path_obj.name, num_tables, num_enhanced,
    )

    # ── Step 9: Compile final document text ───────────────────────────────────
    # General prose comes first, then all enhanced tables appended.
    # If Docling inlines table markdown into export_to_markdown() already,
    # the enhanced blocks serve as corrected replacements at the document tail.
    compiled_text = general_text

    if enhanced_blocks:
        compiled_text += (
            "\n\n"
            "══════════════════════════════════════════════════════════════\n"
            "  FORWARD-FILL ENHANCED TABLES (merged-cell corrected)\n"
            "══════════════════════════════════════════════════════════════\n"
        )
        compiled_text += "".join(enhanced_blocks)

    # ── Step 10: Return as LlamaIndex Document ────────────────────────────────
    return Document(
        text=compiled_text,
        metadata={
            "pdf_path":           str(pdf_path_obj),
            "filename":           pdf_path_obj.name,
            "num_tables":         num_tables,
            "num_tables_enhanced": num_enhanced,
            "source":             "ingest_datasheet_to_llama_document",
        },
    )
