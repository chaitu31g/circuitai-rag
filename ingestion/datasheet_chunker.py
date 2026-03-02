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

from ingestion.vision_processor import VisionProcessor

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
MAX_PROSE_CHARS    = 900     # semantic-pass: max chars per structured chunk
WINDOW_CHARS       = 700     # coverage-pass: characters per sliding window
WINDOW_STEP_CHARS  = 500     # coverage-pass: step size (200 char overlap)

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

def chunk_table(table: dict, section_name: str, part_number: str) -> Optional[Chunk]:
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

    lines = []
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
            parts.append(f"{hdr}={val}")
        if parts:
            lines.append(", ".join(parts))

    if not lines:
        return None

    header_text = " | ".join(h for h in headers if h)
    text = (
        f"{section_name} table for {part_number} [{header_text}]:\n"
        + "\n".join(lines)
    )
    page = (table.get("prov") or [{}])[0].get("page_no")
    return Chunk(
        text=text,
        chunk_type="table",
        metadata={
            "part_number":  part_number,
            "section_name": section_name,
            "page":         page,
            "num_rows":     len(sorted_rows) - 1,
        },
    )


# ---------------------------------------------------------------------------
# Figure / image chunking
# ---------------------------------------------------------------------------

def chunk_figure(figure: dict, part_number: str, pdf_path: Optional[Path] = None) -> Optional[Chunk]:
    # Vision enabled for AWS deployment
    vision = VisionProcessor(model="moondream")

    captions   = figure.get("captions", [])
    cap_text   = " ".join(
        c.get("text", "") if isinstance(c, dict) else str(c)
        for c in captions
    ).strip()
    
    page = (figure.get("prov") or [{}])[0].get("page_no") if figure.get("prov") else None
    bbox = (figure.get("prov") or [{}])[0].get("bbox") if figure.get("prov") else None

    # Pass 1: Identification
    is_graph = "figure" in cap_text.lower() or "curve" in cap_text.lower() or "characteristic" in cap_text.lower()
    
    # Pass 2: Vision Analysis (The "Complete Reading")
    vision_desc = None
    if vision and page and bbox:
        logger.info(f"Analysising figure on page {page} for {part_number}...")
        vision_desc = vision.extract_and_describe(pdf_path, page, bbox, is_graph=is_graph)

    if vision_desc:
        text = f"Visual content description for {part_number} (Page {page}):\n{vision_desc}"
        if cap_text:
            text = f"Figure: {cap_text}\n\nAnalysis:\n{vision_desc}"
    elif cap_text:
        text = f"Figure/Graph for {part_number} (Page {page}): {cap_text}"
    else:
        # Use bbox to make it unique so we don't lose figures without captions
        bbox_str = f"{bbox.get('l', 0):.1f}_{bbox.get('t', 0):.1f}" if bbox else "unknown"
        text = f"Figure/diagram on page {page} for {part_number} [Ref: {bbox_str}]"

    return Chunk(
        text=text,
        chunk_type="figure",
        metadata={
            "part_number":  part_number,
            "section_name": "figures_and_diagrams",
            "page":         page,
            "has_vision":   vision_desc is not None,
            "bbox":         bbox
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


# ---------------------------------------------------------------------------
# Top-level chunker  (dual-pass: semantic + sliding-window coverage)
# ---------------------------------------------------------------------------

def chunk_document(
    docling_data: dict, 
    part_number: Optional[str] = None,
    pdf_path: Optional[Path] = None
) -> list[Chunk]:
    """Chunk an entire parsed datasheet with zero data loss.

    Strategy
    --------
    Pass 1 — Semantic:
        Group text blocks into named sections (features, electrical_characteristics,
        etc.), create smart table chunks, and figure chunks with captions.
        Only ``revision_history`` and ``legal`` sections are skipped.

    Pass 2 — Sliding-window coverage:
        Walk every raw text block that was NOT fully absorbed into a semantic
        chunk and emit overlapping window chunks.  This guarantees that
        badly-formatted PDFs, unusual section names, or edge-case text still
        make it into the vector store.
    """
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

        # Skip purely decorative labels
        if label in _SKIP_LABELS:
            continue

        # Detect section boundary from headers or short text lines
        detected = None
        if label in ("section_header", "title"):
            detected = _detect_section(txt)
        elif label == "text" and len(txt) < 100:
            detected = _detect_section(txt)

        if detected:
            current_section = detected
            # Set retrieval priority
            section_priority.setdefault(detected, {
                "features": 10, "description": 8, "applications": 7,
                "electrical_characteristics": 9, "absolute_maximum_ratings": 9,
            }.get(detected, 3))
            # IMPORTANT: also store the header text itself so "Features" etc. is searchable
            if len(txt) >= MIN_CHUNK_LENGTH and detected not in _SKIP_TYPES:
                sections.setdefault(detected, []).insert(0, txt)
            continue

        # Skip content from sections flagged as useless
        if current_section in _SKIP_TYPES:
            continue

        sections.setdefault(current_section, []).append(txt)
        seen_raws.add(txt)  # mark as semantically absorbed

    # Emit structured chunks
    for sec_name, sec_texts in sections.items():
        priority = section_priority.get(sec_name, 0)
        for chunk in chunk_section(sec_name, sec_texts, part_number, priority):
            _add(chunk)

    # Tables — assign to nearest section by page
    for i, tbl in enumerate(tables):
        page     = (tbl.get("prov") or [{}])[0].get("page_no")
        best_sec = "electrical_characteristics"
        for t in texts:
            t_page = (t.get("prov") or [{}])[0].get("page_no") if t.get("prov") else None
            if t_page == page:
                d = _detect_section(t.get("text", ""))
                if d and d not in _SKIP_TYPES:
                    best_sec = d
                    break
        _add(chunk_table(tbl, best_sec, part_number))

    # Figures — try to find nearby captions if missing
    for fig in pictures:
        if not fig.get("captions"):
            f_prov = (fig.get("prov") or [{}])[0]
            f_page = f_prov.get("page_no")
            f_bbox = f_prov.get("bbox")
            
            if f_page and f_bbox:
                # Search for text blocks on same page that look like captions
                # and haven't been absorbed yet
                best_cap = None
                min_dist = float('inf')
                
                f_y = f_bbox.get("t", 0) # Top coord (assuming BOTTOMLEFT origin is handled)
                
                for t in texts:
                    t_text = t.get("text", "").strip()
                    if not t_text or t_text in seen_raws:
                        continue
                        
                    t_prov = (t.get("prov") or [{}])[0]
                    if t_prov.get("page_no") != f_page:
                        continue
                        
                    # Pattern for common engineering figure labels and typical technical headers
                    is_potential_cap = (
                        re.match(r"^(Figure|Fig|Chart|Graph|Scheme|Diagram|Table)\.?\s*(\d+|[A-Z])", t_text, re.I) or
                        any(kw in t_text for kw in ["Curve", "Characteristic", "Plot", "Waveform", "Diagram", "Circuit"])
                    ) and len(t_text) < 150
                    
                    if is_potential_cap:
                        t_bbox = t_prov.get("bbox")
                        if t_bbox:
                            # Vertical distance check
                            dist = abs(t_bbox.get("t", 0) - f_y)
                            if dist < min_dist:
                                min_dist = dist
                                best_cap = t_text
                
                if best_cap and min_dist < 200:
                    fig["captions"] = [{"text": best_cap}]
                    seen_raws.add(best_cap)

    for fig in pictures:
        _add(chunk_figure(fig, part_number, pdf_path=pdf_path))

    # ═══════════════════════════════════════════════════════════════════════
    # PASS 2 — Sliding-window coverage guarantee
    # ═══════════════════════════════════════════════════════════════════════
    # Collect every raw text block (except noise labels) that was either:
    #   a) in a skipped section and never absorbed, OR
    #   b) not detected by section logic (edge-case text)
    # Then emit overlapping window chunks from the combined text.

    unabsorbed: list[str] = []
    for t in texts:
        label = t.get("label", "text")
        txt   = t.get("text", "").strip()
        if not txt or label in _SKIP_LABELS:
            continue
        if txt in seen_raws:
            continue   # already captured in a semantic chunk
        unabsorbed.append(txt)

    if unabsorbed:
        # Also slide a window over ALL text (not just unabsorbed) for boundary safety
        all_raw = "\n".join(t.get("text", "").strip()
                            for t in texts
                            if t.get("text", "").strip()
                            and t.get("label", "text") not in _SKIP_LABELS)

        # Character-level sliding window
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
        sum(1 for c in all_chunks if c.chunk_type not in ("table", "figure", "raw_text")),
        sum(1 for c in all_chunks if c.chunk_type == "table"),
        sum(1 for c in all_chunks if c.chunk_type == "figure"),
        sum(1 for c in all_chunks if c.chunk_type == "raw_text"),
    )
    return all_chunks
