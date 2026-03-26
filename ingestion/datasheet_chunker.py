"""Datasheet chunker — dual-pass strategy for zero data loss.

Pass 1 (semantic):  Section-aware chunks with rich section labels.
Pass 2 (coverage):  Sliding-window over every raw text block so that
                    nothing ever falls through the cracks.

Both passes feed the same dedup set so the vector store is clean.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ingestion.vision.figure_analyzer import FigureAnalyzer, classify_figure

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text:       str
    chunk_type: str
    metadata:   dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CHUNK_LENGTH   = 10      # characters; below this a chunk is noise
MIN_FIGURE_TEXT    = 20      # characters; below this a figure chunk is discarded
MAX_PROSE_CHARS    = 900     # semantic-pass: max chars per structured chunk
WINDOW_CHARS       = 700     # coverage-pass: characters per sliding window
WINDOW_STEP_CHARS  = 500     # coverage-pass: step size (200 char overlap)

# Patterns that indicate garbage OCR / DePlot output
_FIGURE_GARBAGE_PATTERNS = [
    "TITLE |",          # DePlot corrupted header
    "Seats Bath",       # known OCR corruption
    "electoral divisions",  # OCR gibberish
]

# Canonical section keyword map — order matters, first match wins
_SECTION_MAP = [
    (["feature", "highlights", "key feature"],              "features"),
    (["application"],                                        "applications"),
    (["description", "overview", "general description"],    "description"),

    (["pin configuration", "pin description", "pinning",
      "pin assignment", "pin function"],                     "pin_configuration"),

    (["absolute maximum rating", "maximum rating",
      "limiting value", "stresses above"],                   "absolute_maximum_ratings"),
    (["recommended operating", "operating condition"],       "recommended_operating_conditions"),

    (["electrical characteristic"],                          "electrical_characteristics"),
    (["thermal characteristic", "thermal resistance"],       "thermal_characteristics"),
    (["typical characteristic", "typical performance"],      "typical_characteristics"),

    (["application information", "application note",
      "application circuit", "reference design",
      "typical application"],                                "application_info"),

    (["package", "mechanical", "outline", "dimension"],      "package_info"),
    (["ordering information", "ordering code",
      "device ordering"],                                    "ordering_info"),
    (["marking"],                                            "marking"),
    (["soldering", "solder"],                                "soldering_info"),
    (["quick reference"],                                    "quick_reference"),
    (["test information", "test circuit"],                   "test_info"),
    (["revision history"],                                   "revision_history"),
    (["legal", "disclaimer", "trademark"],                   "legal"),
]

# Truly useless sections for RAG — keep this narrow!
# Everything else (soldering temps, ordering codes, marking) IS useful.
_SKIP_TYPES = {"revision_history", "legal"}

# Docling labels that carry no semantic value
_SKIP_LABELS = {"page_header", "page_footer"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_section(text: str) -> Optional[str]:
    low = text.lower().strip()
    for keywords, section_type in _SECTION_MAP:
        if any(kw in low for kw in keywords):
            return section_type
    return None


def _extract_part_number(texts: list[dict]) -> Optional[str]:
    for t in texts[:20]:
        txt = t.get("text", "").strip()
        if 3 < len(txt) < 40 and re.search(r"[A-Z]{2,}[\d]{2,}", txt):
            return txt
    return None


def _split_prose(text: str, max_len: int = MAX_PROSE_CHARS) -> list[str]:
    """Split long prose at paragraph boundaries without exceeding max_len."""
    if len(text) <= max_len:
        return [text]

    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > max_len:
            chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para) if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Table chunking
# ---------------------------------------------------------------------------

def chunk_table(
    table: dict,
    section_name: str,
    part_number: str,
    table_number: int = 0,
) -> Optional[Chunk]:
    """Convert a parsed Docling table into a semantically rich chunk.

    The output text has two parts:
    1. A natural-language preamble that names the table, section, and all
       parameter names — this is what the embedding model uses to match queries.
    2. The structured data rows in 'Parameter: Value' form so the LLM can
       read exact values without having to decode raw col=val strings.
    """
    cells = table.get("data", {}).get("table_cells", [])
    if not cells:
        return None

    rows_map: dict[int, list] = {}
    for c in cells:
        r = c.get("row_index", c.get("start_row_offset_idx", 0))
        rows_map.setdefault(r, []).append(c)

    sorted_rows = [
        [c.get("text", "").strip()
         for c in sorted(rows_map[r],
                         key=lambda x: x.get("col_index", x.get("start_col_offset_idx", 0)))]
        for r in sorted(rows_map)
    ]

    if len(sorted_rows) < 2:
        return None

    headers  = sorted_rows[0]
    num_cols = len(headers)

    # Build structured data rows in readable form
    data_lines = []
    for row in sorted_rows[1:]:
        while len(row) < num_cols:
            row.append("")
        if not any(cell for cell in row):
            continue
        parts = []
        for i, val in enumerate(row):
            if not val or val == "-":
                continue
            hdr = headers[i] if i < len(headers) and headers[i] else f"col{i}"
            parts.append(f"{hdr}: {val}")
        if parts:
            data_lines.append("  " + " | ".join(parts))

    if not data_lines:
        return None

    # Build semantic preamble — makes embedding match queries about this section
    non_empty_headers = [h for h in headers if h]
    param_names = ", ".join(non_empty_headers[:6])  # first 6 column names
    section_label = section_name.replace("_", " ").title()
    table_label = f"Table {table_number}" if table_number else "Table"

    preamble = (
        f"{table_label}: {section_label} data for {part_number}.\n"
        f"This table lists the following parameters/columns: {param_names}.\n"
        f"It contains {len(data_lines)} row(s) of specification data.\n"
    )

    header_row = " | ".join(h for h in headers if h)
    text = (
        preamble
        + f"\nColumns: {header_row}\n"
        + "\n".join(data_lines)
    )

    page = (table.get("prov") or [{}])[0].get("page_no")
    return Chunk(
        text=text,
        chunk_type="table",
        metadata={
            "part_number":   part_number,
            "section_name":  section_name,
            "table_number":  table_number,
            "table_title":   f"{section_label} — {param_names[:60]}",
            "page":          page,
            "num_rows":      len(sorted_rows) - 1,
            "num_cols":      num_cols,
        },
    )


# ---------------------------------------------------------------------------
# Figure / image chunking
# ---------------------------------------------------------------------------


def chunk_figure(
    figure: dict,
    part_number: str,
    pdf_path: Optional[Path] = None,
    figure_index: int = 0,
) -> Optional[Chunk]:
    """Classify and analyze a figure from a parsed Docling document.

    Routing (Qwen2-VL removed):
      • Graph   → DePlot extracts structured axis/data text.
      • Diagram → Caption-only; Qwen3.5-4B reasons at query time.

    Returns a Chunk with structured metadata matching the target schema:
    {
      "type": "figure",
      "figure_type": "graph" | "diagram",
      "component": <part_number>,
      "page": <int>,
      "title": <chart title extracted by DePlot or caption>,
      "x_axis": <x-axis label>,
      "y_axis": <y-axis label>,
      "text": <full chunk text for embedding>,
    }

    Parameters
    ----------
    figure_index : Global sequential index within the document.
                  Embedded into fallback chunk text to guarantee uniqueness
                  even when a figure has no caption and no bbox.
    """
    analyzer = FigureAnalyzer()

    captions = figure.get("captions", [])
    cap_text = " ".join(
        c.get("text", "") if isinstance(c, dict) else str(c)
        for c in captions
    ).strip()

    page = (figure.get("prov") or [{}])[0].get("page_no") if figure.get("prov") else None
    bbox = (figure.get("prov") or [{}])[0].get("bbox") if figure.get("prov") else None

    # ── Run vision pipeline (DePlot for graphs, caption-only for diagrams) ────
    vision_text = None
    if page and bbox:
        logger.info(
            "FigureAnalyzer: dispatching figure %d page=%s for %s  caption=%r",
            figure_index, page, part_number, cap_text[:60],
        )
        vision_text = analyzer.extract_and_describe(
            pdf_path=pdf_path,
            page_no=page,
            bbox=bbox,
            caption=cap_text,
            part_number=part_number,
        )

    # ── Determine figure_type from FigureAnalyzer output header ───────────────
    # FigureAnalyzer always prefixes output with "Figure Type: Graph" / "Diagram".
    if vision_text:
        first_line = (vision_text.splitlines() or [""])[0].lower()
        if "graph" in first_line:
            figure_type = "graph"
        elif "diagram" in first_line:
            figure_type = "diagram"
        else:
            figure_type = "unknown"
    else:
        # Caption-based fallback using the shared keyword classifier.
        figure_type = classify_figure(caption=cap_text)

    # ── Extract chart title, axes from DePlot vision text ────────────────────
    title  = cap_text or ""
    x_axis = ""
    y_axis = ""
    if vision_text:
        for line in vision_text.splitlines():
            ll = line.lower().strip()
            if ll.startswith("chart title:"):
                title = line.split(":", 1)[-1].strip() or title
            elif ll.startswith("x-axis:") or ll.startswith("x axis:"):
                x_axis = line.split(":", 1)[-1].strip()
            elif ll.startswith("y-axis:") or ll.startswith("y axis:"):
                y_axis = line.split(":", 1)[-1].strip()

    # ── Build chunk text ──────────────────────────────────────────────────────
    if vision_text:
        text = vision_text
    elif cap_text:
        type_label = "Graph" if figure_type == "graph" else "Diagram"
        text = (
            f"Figure Type: {type_label}\n"
            f"Caption: {cap_text}\n"
            f"Component: {part_number}\n"
            f"Page: {page}\n"
            f"Figure Index: {figure_index}\n\n"
            f"Description: {cap_text}"
        )
    else:
        bbox_str = f"{bbox.get('l', 0):.1f}_{bbox.get('t', 0):.1f}" if bbox else "unknown"
        type_label = "Graph" if figure_type == "graph" else "Diagram"
        text = (
            f"Figure Type: {type_label}\n"
            f"Component: {part_number}\n"
            f"Page: {page}\n"
            f"Figure Index: {figure_index}\n"
            f"Ref: {bbox_str}"
        )

    # ── Quality gate: discard low-quality / corrupted figure chunks ──────────
    text_for_quality_check = text.strip()
    is_low_quality = (
        len(text_for_quality_check) < MIN_FIGURE_TEXT
        or any(pat in text_for_quality_check for pat in _FIGURE_GARBAGE_PATTERNS)
    )
    if is_low_quality:
        logger.warning(
            "Skipping low-quality figure chunk (index=%d, part=%s, text_preview=%r)",
            figure_index, part_number, text_for_quality_check[:80],
        )
        return None

    return Chunk(
        text=text,
        chunk_type="figure",
        metadata={
            # Standard type markers (both fields for backward compat)
            "type":          "figure",
            "chunk_type":    "figure",
            # Figure classification
            "figure_type":   figure_type,
            "figure_index":  figure_index,
            # Component provenance
            "part_number":   part_number,
            "component":     part_number,
            "section_name":  "figures_and_diagrams",
            "page":          page,
            # Content fields — improves embedding quality & retrieval accuracy
            "title":         title,
            "caption":       cap_text or "",
            "text":          cap_text or "",
            "x_axis":        x_axis,
            "y_axis":        y_axis,
            "has_vision":    vision_text is not None,
        },
    )


# ---------------------------------------------------------------------------
# Prose section chunking
# ---------------------------------------------------------------------------

def chunk_section(
    section_name: str,
    section_texts: list[str],
    part_number: str,
    priority: int = 0,
) -> list[Chunk]:
    if not section_texts:
        return []

    merged = "\n".join(t.strip() for t in section_texts if t.strip())
    if len(merged) < MIN_CHUNK_LENGTH:
        return []

    parts  = _split_prose(merged)
    result = []
    for i, part in enumerate(parts):
        meta: dict = {
            "part_number":  part_number,
            "section_name": section_name,
            "priority":     priority,
        }
        if len(parts) > 1:
            meta["subsection"] = i + 1
        result.append(Chunk(
            text=f"{section_name} of {part_number}: {part}",
            chunk_type=section_name,
            metadata=meta,
        ))
    return result


def _get_table_contexts(docling_data: dict) -> dict:
    """Refined Context Tracking utilizing Docling's 'body' hierarchy with spatial fallback.
    Returns: table_index -> (section_name, table_title)
    """
    body = docling_data.get("body", {})
    children = body.get("children", [])
    texts = docling_data.get("texts", [])
    groups = docling_data.get("groups", [])
    tables = docling_data.get("tables", [])
    
    table_contexts = {}
    current_section = "description"
    last_heading = "Table"
    
    # 1. Hierarchy Pass
    def traverse(ref):
        nonlocal current_section, last_heading
        if not isinstance(ref, str) or not ref.startswith("#/"):
            return
        parts = ref.split("/")
        if len(parts) < 3: return
        etype = parts[1]
        try:
            eidx = int(parts[2])
        except: return
        
        if etype == "texts" and eidx < len(texts):
            t = texts[eidx]
            txt = t.get("text", "").strip()
            label = t.get("label", "text").lower()
            det = _detect_section(txt)
            if det and any(h in label for h in ("header", "title", "heading")):
                current_section = det
                last_heading = txt
            elif any(h in label for h in ("header", "title", "heading", "caption")):
                last_heading = txt
            elif "Table" in txt and len(txt) < 120:
                last_heading = txt
        elif etype == "tables":
            table_contexts[eidx] = (current_section, last_heading)
        elif etype == "groups" and eidx < len(groups):
            for child in groups[eidx].get("children", []):
                cref = child.get("$ref")
                if cref: traverse(cref)

    for child in children:
        ref = child.get("$ref")
        if ref: traverse(ref)

    # 2. Spatial Fallback for any disconnected tables
    for i in range(len(tables)):
        if i in table_contexts: continue
        
        tbl = tables[i]
        t_prov = (tbl.get("prov") or [{}])[0]
        t_page = t_prov.get("page_no")
        t_bbox = t_prov.get("bbox", {})
        t_y = t_bbox.get("t", 0) if isinstance(t_bbox, dict) else 0
        
        best_sec = "description"
        best_title = "Table"
        min_dist = float('inf')
        
        for t in texts:
            prov = (t.get("prov") or [{}])[0]
            page = prov.get("page_no")
            if page is None or t_page is None or page > t_page: continue
            
            # If same page, must be above
            bbox = prov.get("bbox", {})
            y = bbox.get("t", 0) if isinstance(bbox, dict) else 0
            if page == t_page and y >= t_y and t_y > 0: continue
            
            txt = t.get("text", "").strip()
            label = t.get("label", "text").lower()
            
            # Track most recent section
            det = _detect_section(txt)
            if det and any(h in label for h in ("header", "title", "heading")):
                best_sec = det
            
            # Check if this could be the title
            if any(h in label for h in ("header", "title", "heading", "caption")):
                dist = (t_page - page) * 2000 + (t_y - y)
                if dist < min_dist:
                    min_dist = dist
                    best_title = txt
                    
        table_contexts[i] = (best_sec, best_title)
            
    return table_contexts


def chunk_document(
    docling_data: dict, 
    part_number: Optional[str] = None,
    pdf_path: Optional[Path] = None
) -> list[Chunk]:
    """Chunk an entire parsed datasheet with zero data loss."""
    texts    = docling_data.get("texts",    [])
    tables   = docling_data.get("tables",   [])
    pictures = docling_data.get("pictures", [])

    if not part_number:
        part_number = _extract_part_number(texts) or "unknown"

    all_chunks:  list[Chunk] = []
    seen_texts:  set[str]   = set()   # dedup on full chunk text
    seen_raws:   set[str]   = set()   # dedup on raw source text

    def _add(chunk: Optional[Chunk]) -> None:
        if chunk is None:
            return
        key = chunk.text.strip()
        if len(key) < MIN_CHUNK_LENGTH:
            return
        if key in seen_texts:
            return
        seen_texts.add(key)
        all_chunks.append(chunk)

    def _add_figure(chunk: Optional[Chunk]) -> None:
        if chunk is None:
            return
        key = chunk.text.strip()
        if not key:
            return
        if key in seen_texts:
            return
        seen_texts.add(key)
        all_chunks.append(chunk)

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 1 — Semantic chunking
    # ═══════════════════════════════════════════════════════════════════════

    current_section = "description"
    sections: dict[str, list[str]] = {}
    section_priority: dict[str, int] = {}

    for t in texts:
        label = t.get("label", "text")
        txt   = t.get("text", "").strip()
        if not txt:
            continue

        if label in _SKIP_LABELS:
            continue

        detected = None
        if label in ("section_header", "title"):
            detected = _detect_section(txt)
        elif label == "text" and len(txt) < 100:
            detected = _detect_section(txt)

        if detected:
            current_section = detected
            section_priority.setdefault(detected, {
                "features": 10, "description": 8, "applications": 7,
                "electrical_characteristics": 9, "absolute_maximum_ratings": 9,
            }.get(detected, 3))
            if len(txt) >= MIN_CHUNK_LENGTH and detected not in _SKIP_TYPES:
                sections.setdefault(detected, []).insert(0, txt)
            continue

        if current_section in _SKIP_TYPES:
            continue

        sections.setdefault(current_section, []).append(txt)
        seen_raws.add(txt)

    for sec_name, sec_texts in sections.items():
        priority = section_priority.get(sec_name, 0)
        for chunk in chunk_section(sec_name, sec_texts, part_number, priority):
            _add(chunk)

    # Tables — Use document hierarchy for titles and sections
    has_valid_pdf = pdf_path and Path(pdf_path).exists()
    
    if has_valid_pdf:
        from rag_pipeline.parsers.hybrid_table_parser_v3 import extract_tables_hybrid_v3
        try:
            table_chunks = extract_tables_hybrid_v3(str(pdf_path), docling_data, part_number)
            for tc in table_chunks:
                _add(tc)
        except Exception as e:
            logger.error(f"Hybrid table parser completely failed for {part_number}: {e}")
            has_valid_pdf = False  # trigger fallback

    if not has_valid_pdf:
        # Fallback to standard docling-only extraction
        table_contexts = _get_table_contexts(docling_data)
        
        for i, tbl in enumerate(tables):
            ctx = table_contexts.get(i)
            if ctx:
                best_sec, table_title = ctx
            else:
                # Fallback to page-based logic if hierarchy failed
                page     = (tbl.get("prov") or [{}])[0].get("page_no")
                best_sec = "electrical_characteristics"
                for t in texts:
                    t_page = (t.get("prov") or [{}])[0].get("page_no") if t.get("prov") else None
                    if t_page == page:
                        d = _detect_section(t.get("text", ""))
                        if d and d not in _SKIP_TYPES:
                            best_sec = d
                            break
                table_title = ""
    
            from rag_pipeline.utils.parameter_extractor import extract_parameter_rows
            row_chunks = extract_parameter_rows(tbl, best_sec, part_number, i + 1, table_title=table_title)
            for rc in row_chunks:
                _add(rc)

    # Figures
    for fig in pictures:
        if not fig.get("captions"):
            f_prov = (fig.get("prov") or [{}])[0]
            f_page = f_prov.get("page_no")
            f_bbox = f_prov.get("bbox")
            
            if f_page and f_bbox:
                best_cap = None
                min_dist = float('inf')
                f_y = f_bbox.get("t", 0)
                
                for t in texts:
                    t_text = t.get("text", "").strip()
                    if not t_text or t_text in seen_raws:
                        continue
                    t_prov = (t.get("prov") or [{}])[0]
                    if t_prov.get("page_no") != f_page:
                        continue
                    is_potential_cap = (
                        re.match(r"^(Figure|Fig|Chart|Graph|Scheme|Diagram|Table)\.?\s*(\d+|[A-Z])", t_text, re.I) or
                        any(kw in t_text for kw in ["Curve", "Characteristic", "Plot", "Waveform", "Diagram", "Circuit"])
                    ) and len(t_text) < 150
                    if is_potential_cap:
                        t_bbox = t_prov.get("bbox")
                        if t_bbox:
                            dist = abs(t_bbox.get("t", 0) - f_y)
                            if dist < min_dist:
                                min_dist = dist
                                best_cap = t_text
                if best_cap and min_dist < 200:
                    fig["captions"] = [{"text": best_cap}]
                    seen_raws.add(best_cap)

    for fig_idx, fig in enumerate(pictures):
        _add_figure(chunk_figure(fig, part_number, pdf_path=pdf_path, figure_index=fig_idx))

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 2 — Sliding-window coverage guarantee
    # ═══════════════════════════════════════════════════════════════════════
    unabsorbed: list[str] = []
    for t in texts:
        label = t.get("label", "text")
        txt   = t.get("text", "").strip()
        if not txt or label in _SKIP_LABELS:
            continue
        if txt in seen_raws:
            continue
        unabsorbed.append(txt)

    if unabsorbed:
        all_raw = "\n".join(t.get("text", "").strip()
                            for t in texts
                            if t.get("text", "").strip()
                            and t.get("label", "text") not in _SKIP_LABELS)
        pos = 0
        win_idx = 0
        while pos < len(all_raw):
            snippet = all_raw[pos : pos + WINDOW_CHARS].strip()
            if len(snippet) >= MIN_CHUNK_LENGTH:
                _add(Chunk(
                    text=f"datasheet text of {part_number} (window {win_idx}): {snippet}",
                    chunk_type="raw_text",
                    metadata={
                        "part_number":  part_number,
                        "section_name": "raw_coverage",
                        "window_index": win_idx,
                    },
                ))
            pos     += WINDOW_STEP_CHARS
            win_idx += 1

    logger.info(
        "Chunked %s: %d total chunks — %d semantic, %d tables, %d figures, %d coverage windows",
        part_number, len(all_chunks),
        sum(1 for c in all_chunks if c.chunk_type not in ("table", "parameter_row", "figure", "raw_text")),
        sum(1 for c in all_chunks if c.chunk_type in ("table", "parameter_row")),
        sum(1 for c in all_chunks if c.chunk_type == "figure"),
        sum(1 for c in all_chunks if c.chunk_type == "raw_text"),
    )
    return all_chunks
