"""
rag_pipeline/utils/table_aware_ingest.py
──────────────────────────────────────────────────────────────────────────────
Drop-in integration layer that enriches Docling-parsed datasheets with
per-row table chunks **without modifying any existing function**.

How it works
────────────
1.  Call the existing ``chunk_document()`` as usual.
2.  Additionally, iterate over every table in the parsed JSON, calling the
    new ``format_table_rows()`` utility to produce one readable text chunk
    per data row.
3.  De-duplicate against chunks already emitted by ``chunk_document()`` so
    no text appears twice.
4.  Append the enriched row-chunks to the chunk list.
5.  Return a flat list of dicts matching the schema expected by
    ``EmbeddingPipeline.run()``.

Nothing in the original pipeline (``datasheet_chunker.py``,
``embed_pipeline.py``, ``chroma_store.py``) is modified in any way.

Usage
─────
Replace the ``chunk_document()`` call in your ingestion script with
``table_aware_chunk_document()``, or call ``enrich_with_table_rows()``
after you already have the chunk list.

    from rag_pipeline.utils.table_aware_ingest import table_aware_chunk_document

    chunks = table_aware_chunk_document(docling_data, part_number, pdf_path)
    # → list of dicts ready for EmbeddingPipeline.run()

Alternatively, integrate inside the existing run_ingest.py without touching it
by calling this module as a pre-processing step before EmbeddingPipeline.run().
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestion.datasheet_chunker import chunk_document
from rag_pipeline.utils.table_formatter import format_table_rows

logger = logging.getLogger(__name__)

# Minimum character length for a table row-chunk to be kept
_MIN_CHUNK_LEN = 20


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_to_dict(chunk) -> Dict[str, Any]:
    """Convert a ``Chunk`` dataclass instance to an ingestion-compatible dict.

    Works whether the object is a proper dataclass or already a dict
    (defensive — makes the function usable from any call site).
    """
    if isinstance(chunk, dict):
        return chunk
    if dataclasses.is_dataclass(chunk):
        return dataclasses.asdict(chunk)
    # Fallback — try attribute access
    return {
        "text":       getattr(chunk, "text", ""),
        "chunk_type": getattr(chunk, "chunk_type", "unknown"),
        "metadata":   getattr(chunk, "metadata", {}),
    }


def _detect_best_section(table: Dict[str, Any], texts: List[Dict[str, Any]]) -> str:
    """Return the most likely section label for a given Docling table dict.

    Mirrors the logic inside ``chunk_document()`` without duplicating its
    private helpers — looks for a text block on the same page whose content
    matches a known section keyword.
    """
    from ingestion.datasheet_chunker import _detect_section  # private but stable
    page = (table.get("prov") or [{}])[0].get("page_no")
    for t in texts:
        t_page = (t.get("prov") or [{}])[0].get("page_no") if t.get("prov") else None
        if t_page == page:
            detected = _detect_section(t.get("text", ""))
            if detected:
                return detected
    return "electrical_characteristics"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def enrich_with_table_rows(
    base_chunks: List[Any],
    docling_data: Dict[str, Any],
    part_number: str = "",
) -> List[Dict[str, Any]]:
    """Append per-row table chunks to an existing chunk list.

    This is the core enrichment step.  Call it *after* ``chunk_document()``
    to add fine-grained parameter rows without re-running the full semantic
    pass.

    Parameters
    ----------
    base_chunks:
        Output of ``chunk_document()`` — list of ``Chunk`` dataclass instances
        or dicts.
    docling_data:
        The original parsed Docling JSON dict (contains ``"tables"`` key).
    part_number:
        Component identifier, e.g. "IRF540N".

    Returns
    -------
    list[dict]
        ``base_chunks`` converted to dicts **plus** all new row-level chunks,
        de-duplicated by text content.
    """
    texts  = docling_data.get("texts",  [])
    tables = docling_data.get("tables", [])

    # Convert existing chunks to dicts and build dedup set
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for c in base_chunks:
        d = _chunk_to_dict(c)
        key = d.get("text", "").strip()
        if key and key not in seen:
            seen.add(key)
            result.append(d)

    # Generate enriched row-level chunks for every table
    new_row_chunks = 0
    for i, table in enumerate(tables):
        section = _detect_best_section(table, texts)
        row_texts = format_table_rows(
            table_data=table,
            section_name=section,
            part_number=part_number,
            table_number=i + 1,
        )

        for row_text in row_texts:
            key = row_text.strip()
            if not key or len(key) < _MIN_CHUNK_LEN or key in seen:
                continue
            seen.add(key)
            result.append({
                "text":       row_text,
                "chunk_type": "table_row",
                "metadata": {
                    "part_number":  part_number,
                    "section_name": section,
                    "table_number": i + 1,
                    "chunk_source": "table_formatter",
                },
            })
            new_row_chunks += 1

    logger.info(
        "enrich_with_table_rows: added %d row-level chunks to %d base chunks "
        "(total: %d) for part=%s",
        new_row_chunks, len(base_chunks), len(result), part_number,
    )
    return result


def table_aware_chunk_document(
    docling_data: Dict[str, Any],
    part_number: Optional[str] = None,
    pdf_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Full replacement for ``chunk_document()`` with table-row enrichment.

    Calls the original ``chunk_document()`` unchanged, then enriches the
    output with per-row table chunks from ``format_table_rows()``.

    Parameters
    ----------
    docling_data:
        Parsed Docling output dict.
    part_number:
        Optional component identifier.  Auto-detected if not supplied.
    pdf_path:
        Path to the source PDF, forwarded to ``chunk_document()`` for
        figure extraction.

    Returns
    -------
    list[dict]
        All chunks (semantic + table rows) as plain dicts, ready for
        ``EmbeddingPipeline.run()``.

    Example
    -------
    Replace::

        from ingestion.datasheet_chunker import chunk_document
        chunks = chunk_document(docling_data, part_number, pdf_path)

    With::

        from rag_pipeline.utils.table_aware_ingest import table_aware_chunk_document
        chunks = table_aware_chunk_document(docling_data, part_number, pdf_path)
    """
    # Step 1: run the original chunker (unchanged)
    base_chunks = chunk_document(
        docling_data=docling_data,
        part_number=part_number,
        pdf_path=pdf_path,
    )

    # Resolve part_number the same way chunk_document() does if not passed
    from ingestion.datasheet_chunker import _extract_part_number
    if not part_number:
        part_number = _extract_part_number(docling_data.get("texts", [])) or "unknown"

    # Step 2: enrich with row-level table chunks
    return enrich_with_table_rows(base_chunks, docling_data, part_number=part_number)
