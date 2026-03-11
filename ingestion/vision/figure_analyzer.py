"""
ingestion/vision/figure_analyzer.py
────────────────────────────────────────────────────────────────────────────
Vision analyzer for electronics datasheet figures.

Architecture (Qwen2-VL removed, DePlot only)
─────────────────────────────────────────────
  Caption/keyword classifier
        │
        ▼
  figure_type = "graph" │ "diagram"
        │
        ├─ "graph"   → DePlot  (google/deplot)  extracts structured chart data
        └─ "diagram" → caption-only  (no vision model needed)

Diagrams (circuit drawings, package outlines, block diagrams) are now
described using only their caption text. Qwen3.5-4B in the RAG chat layer
provides any further reasoning or interpretation when a user asks about them.

This removes the Qwen2-VL-2B-Instruct dependency entirely, which was the
main source of GPU OOM errors on Colab T4s when DePlot was also loaded.

Singleton pattern
─────────────────
DePlot is loaded exactly ONCE at process startup (thread-safe).
Every call to FigureAnalyzer.analyze() reuses the shared instance.

Output format
─────────────
Regardless of which path was taken, analyze() returns a structured
string that becomes the chunk text stored in ChromaDB:

  Figure Type: Graph          ← or "Diagram"
  Caption: <caption text>
  Component: <part_number>
  Page: <n>

  X-axis: ...                 ← DePlot path (graphs only)
  Y-axis: ...
  Extracted Data:
    25°C → 3.0 A
    ...
  Trend: ...

  ── or ──

  Figure Type: Diagram        ← caption-only path
  Caption: <caption text>
  Component: <part_number>
  Page: <n>

  Description: <caption repeated for embedding quality>
"""

from __future__ import annotations

import logging
import re
import threading
from io import BytesIO
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ── Model identifiers ─────────────────────────────────────────────────────────
_DEPLOT_MODEL_ID = "google/deplot"

# ── Module-level singletons ───────────────────────────────────────────────────
_deplot_processor = None
_deplot_model     = None

_device    = None        # 'cuda' or 'cpu'
_load_lock = threading.Lock()

# ── Graph-detection keyword set ───────────────────────────────────────────────
# Only phrases that are unambiguously associated with axis-bearing charts/plots.
# Generic single-word terms like 'output', 'resistance', 'temperature'
# are intentionally excluded — they also appear in table captions and would
# incorrectly route table images to DePlot.
_GRAPH_CAPTION_KEYWORDS = {
    "graph", "curve", "plot", "vs.", "versus",
    "power derating", "gate charge", "output characteristic",
    "transfer characteristic", "safe operating area", "soa",
    "switching waveform", "capacitance vs", "impedance vs",
    "frequency response", "temperature coefficient",
    "drain current vs", "power dissipation vs",
    # Extra broad keywords requested for improved recall
    "vs", "characteristic", "temperature", "current", "voltage", "power",
}


# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (DePlot only — called once, thread-safe)
# ═════════════════════════════════════════════════════════════════════════════

def _load_deplot() -> None:
    """Load DePlot singleton on demand. No-op on subsequent calls."""
    global _deplot_processor, _deplot_model
    global _device

    with _load_lock:
        if _deplot_model is not None:
            return   # already loaded

        try:
            import torch
            from transformers import AutoProcessor, Pix2StructForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Required packages missing. Run:\n"
                "  pip install transformers>=4.45 accelerate pillow"
            ) from exc

        _device = "cuda" if torch.cuda.is_available() else "cpu"

        # DePlot's processor outputs float32 tensors regardless of device.
        # Always load in float32 to avoid dtype mismatch on generate().
        logger.info("Loading DePlot: %s  (device=%s, dtype=float32) …", _DEPLOT_MODEL_ID, _device)
        try:
            _deplot_processor = AutoProcessor.from_pretrained(_DEPLOT_MODEL_ID)
            _deplot_model = Pix2StructForConditionalGeneration.from_pretrained(
                _DEPLOT_MODEL_ID,
                torch_dtype=torch.float32,   # must match processor output dtype
            )
            _deplot_model.to(_device)
            _deplot_model.eval()
            logger.info(
                "DePlot loaded ✓  (device=%s, dtype=%s)",
                _device, _deplot_model.dtype,
            )
        except Exception as exc:
            logger.warning(
                "DePlot failed to load — graphs will fall back to caption-only. Error: %s", exc
            )
            _deplot_processor = None
            _deplot_model     = None


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE TYPE CLASSIFICATION  (caption keyword + pixel heuristic)
# ═════════════════════════════════════════════════════════════════════════════

def classify_figure(caption: str = "", image_bytes: Optional[bytes] = None) -> str:
    """Classify a figure as either 'graph' or 'diagram'.

    Parameters
    ----------
    caption     : Caption text extracted from the PDF (may be empty).
    image_bytes : Raw PNG bytes of the figure (optional; used for pixel heuristic).

    Returns
    -------
    'graph'   — axis-bearing chart/plot; route to DePlot.
    'diagram' — circuit drawing, package outline, block diagram, table image;
                caption-only description.
    """
    # 1. Caption keyword check (fast path — no image decoding).
    cap_lower = caption.lower()
    if any(kw in cap_lower for kw in _GRAPH_CAPTION_KEYWORDS):
        logger.debug("classify_figure → graph  (caption keyword match: %r)", caption[:60])
        return "graph"

    # 2. Pixel heuristic (only if image bytes provided).
    if image_bytes:
        try:
            from PIL import Image
            import numpy as np

            img = Image.open(BytesIO(image_bytes)).convert("L")   # grayscale

            max_w = 400
            if img.width > max_w:
                scale = max_w / img.width
                img   = img.resize((max_w, int(img.height * scale)), Image.LANCZOS)

            arr  = np.array(img, dtype=np.float32)
            dark = (arr < 128).astype(np.uint8)
            h, w = dark.shape

            row_sum = dark.sum(axis=1)
            col_sum = dark.sum(axis=0)

            long_h_lines = int((row_sum > 0.60 * w).sum())
            long_v_lines = int((col_sum > 0.60 * h).sum())

            if long_h_lines >= 5 or long_v_lines >= 4:
                logger.debug(
                    "classify_figure → graph  (pixel heuristic: h=%d v=%d)",
                    long_h_lines, long_v_lines,
                )
                return "graph"

            # Edge-density heuristic
            edge_rows  = max(1, h // 8)
            edge_cols  = max(1, w // 8)
            top_strip  = float(dark[:edge_rows, :].mean())
            bot_strip  = float(dark[h - edge_rows:, :].mean())
            left_strip = float(dark[:, :edge_cols].mean())
            rgt_strip  = float(dark[:, w - edge_cols:].mean())

            edge_density = top_strip + bot_strip + left_strip + rgt_strip
            if edge_density > 0.65:
                logger.debug(
                    "classify_figure → graph  (edge-density: %.3f)", edge_density
                )
                return "graph"

        except Exception as exc:
            logger.debug("Pixel-heuristic failed (continuing): %s", exc)

    logger.debug("classify_figure → diagram  (no graph features detected)")
    return "diagram"


# ═════════════════════════════════════════════════════════════════════════════
# DEPLOT PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

_DEPLOT_PROMPT = "Generate the underlying data table of the figure below:"

def _run_deplot(image_bytes: bytes) -> Optional[str]:
    """Send image to DePlot and return linearised chart data as text.

    Optimisations vs previous version:
      • Images resized to 512 × 512 before inference (saves GPU memory)
      • torch.cuda.empty_cache() called before each run
      • DePlot loaded in float32 to avoid dtype mismatch

    DePlot output is already structured (e.g. 'TITLE | x1 | x2 / y1 | v1 | v2'),
    which we reformat into a human-readable block for chunking.
    """
    if _deplot_model is None or _deplot_processor is None:
        return None

    try:
        import torch
        from PIL import Image

        # ── GPU memory cleanup before inference ───────────────────────────────
        if _device == "cuda":
            torch.cuda.empty_cache()

        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        # ── Resize to 512×512 to reduce GPU memory pressure ───────────────────
        img = img.resize((512, 512), Image.LANCZOS)

        logger.info("Running DePlot inference on %dx%d image …", *img.size)

        # Step 1: encode
        inputs = _deplot_processor(
            images=img,
            text=_DEPLOT_PROMPT,
            return_tensors="pt",
        )

        # Step 2: move to device
        inputs = {k: v.to(_deplot_model.device) for k, v in inputs.items()}

        # Step 3: cast floats to model dtype (belt-and-suspenders)
        model_dtype = _deplot_model.dtype
        inputs = {
            k: v.to(dtype=model_dtype) if (hasattr(v, "dtype") and v.is_floating_point()) else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            generated_ids = _deplot_model.generate(
                **inputs,
                max_new_tokens=512,
            )

        raw = _deplot_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        # ── GPU cleanup after inference ────────────────────────────────────────
        if _device == "cuda":
            torch.cuda.empty_cache()

        logger.info("DePlot extracted chart data successfully")
        return _format_deplot_output(raw)

    except Exception as exc:
        logger.error("DePlot inference failed: %s", exc, exc_info=True)
        if _device == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        return None


def _format_deplot_output(raw: str) -> str:
    """Convert DePlot's LINEARIZED TABLE format to a readable chunk string.

    DePlot typically outputs:
        TITLE | Col1 | Col2 / Row1 | v1 | v2 / Row2 | v3 | v4

    We split on ' / ' and ' | ' to produce:
        Extracted Data:
          Row1 | Col1 → v1
          Row1 | Col2 → v2
          ...
    """
    if not raw:
        return ""

    lines = [raw]   # default: keep raw if parsing fails

    try:
        rows = [r.strip() for r in raw.split(" / ") if r.strip()]
        if len(rows) >= 2:
            header_cols = [c.strip() for c in rows[0].split(" | ")]
            title       = header_cols[0] if header_cols else "Unknown"
            col_names   = header_cols[1:] if len(header_cols) > 1 else []

            data_lines = []
            for row in rows[1:]:
                cells = [c.strip() for c in row.split(" | ")]
                row_label = cells[0] if cells else ""
                for i, val in enumerate(cells[1:]):
                    col = col_names[i] if i < len(col_names) else f"col{i+1}"
                    if val:
                        data_lines.append(f"  {row_label} | {col} → {val}")

            trend = _infer_trend(rows[1:])

            lines = [
                f"Chart Title: {title}",
                "",
                "Extracted Data:",
            ] + data_lines

            if trend:
                lines += ["", f"Trend: {trend}"]

    except Exception:
        pass   # fall back to raw string

    return "\n".join(lines)


def _infer_trend(data_rows: list[str]) -> str:
    """Very lightweight trend inference from DePlot row strings."""
    try:
        values = []
        for row in data_rows:
            cells = row.split(" | ")
            if len(cells) >= 2:
                for cell in cells[1:]:
                    m = re.search(r"[-+]?\d+\.?\d*", cell)
                    if m:
                        values.append(float(m.group()))

        if len(values) >= 2:
            if values[-1] > values[0] * 1.05:
                return "Values increase across the range."
            elif values[-1] < values[0] * 0.95:
                return "Values decrease across the range."
            else:
                return "Values remain approximately constant across the range."
    except Exception:
        pass
    return ""


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE — FigureAnalyzer
# ═════════════════════════════════════════════════════════════════════════════

class FigureAnalyzer:
    """Figure analysis with caption-based classifier and DePlot for graphs.

    Routing:
      • Graph  → DePlot linearises the chart into structured axis/data text.
      • Diagram → Caption text only (no vision model required).

    Qwen2-VL-2B has been removed entirely. Diagram descriptions are now
    generated from the caption alone; Qwen3.5-4B in the RAG layer handles
    any further reasoning when users ask about diagrams.

    Usage (same API as before — callers need no changes):

        analyzer = FigureAnalyzer()
        description = analyzer.extract_and_describe(
            pdf_path, page_no, bbox,
            caption="Drain Current vs Temperature",
            part_number="BSS84P",
        )
    """

    def __init__(self) -> None:
        try:
            _load_deplot()
        except Exception as exc:
            logger.warning(
                "FigureAnalyzer: DePlot failed to load — "
                "all figures will use caption-only mode. Error: %s", exc,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        image_bytes: bytes,
        caption: str = "",
        part_number: str = "unknown",
        page: Optional[int] = None,
    ) -> str:
        """Classify and analyze a figure, returning structured chunk text.

        Parameters
        ----------
        image_bytes : Raw PNG bytes of the figure region.
        caption     : Caption text from PDF (may be empty).
        part_number : Component identifier (e.g. 'BSS84P').
        page        : Source PDF page number (1-indexed).

        Returns
        -------
        Structured string ready for embedding.
        """
        figure_type = classify_figure(caption=caption, image_bytes=image_bytes)
        logger.info(
            "FigureAnalyzer: page=%s  type=%s  part=%s  caption=%r",
            page, figure_type, part_number, caption[:60],
        )

        vision_text: Optional[str] = None

        if figure_type == "graph":
            vision_text = _run_deplot(image_bytes)
            if vision_text is None:
                logger.warning(
                    "DePlot returned None for graph on page %s — using caption-only fallback", page
                )
                # No Qwen2-VL fallback: remain caption-only
        # diagrams: caption-only — no vision model called

        return self._build_chunk_text(
            figure_type=figure_type,
            caption=caption,
            part_number=part_number,
            page=page,
            vision_text=vision_text,
        )

    def extract_and_describe(
        self,
        pdf_path: Path,
        page_no: int,
        bbox: dict,
        caption: str = "",
        part_number: str = "unknown",
    ) -> str:
        """Crop a figure from a PDF page and run analyze() on the bytes.

        Parameters
        ----------
        pdf_path    : Absolute path to the source PDF.
        page_no     : 1-indexed page number (Docling convention).
        bbox        : Docling bbox dict — keys: l, t, r, b, coord_origin.
        caption     : Figure caption text.
        part_number : Component identifier.

        Returns
        -------
        Structured chunk string (may be caption-only if image extraction fails).
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.warning("FigureAnalyzer: PDF not found at %s", pdf_path)
                return self._build_chunk_text("unknown", caption, part_number, page_no, None)

            with fitz.open(str(pdf_path)) as doc:
                page     = doc[page_no - 1]   # 1-indexed → 0-indexed
                p_height = page.rect.height

                l = bbox.get("l", 0)
                t = bbox.get("t", 0)
                r = bbox.get("r", 0)
                b = bbox.get("b", 0)

                # Docling BOTTOMLEFT → PyMuPDF TOPLEFT
                if bbox.get("coord_origin") == "BOTTOMLEFT":
                    rect = fitz.Rect(l, p_height - t, r, p_height - b)
                else:
                    rect = fitz.Rect(l, t, r, b)

                # Small padding + 2× super-sampling for model clarity
                rect.x0 -= 2; rect.y0 -= 2
                rect.x1 += 2; rect.y1 += 2

                pix         = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
                image_bytes = pix.tobytes("png")

            return self.analyze(
                image_bytes=image_bytes,
                caption=caption,
                part_number=part_number,
                page=page_no,
            )

        except Exception as exc:
            logger.error("FigureAnalyzer.extract_and_describe failed: %s", exc, exc_info=True)
            return self._build_chunk_text("unknown", caption, part_number, page_no, None)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_chunk_text(
        figure_type: str,
        caption: str,
        part_number: str,
        page: Optional[int],
        vision_text: Optional[str],
    ) -> str:
        """Assemble the structured chunk string for ChromaDB storage."""
        type_label = {
            "graph":   "Graph",
            "diagram": "Diagram",
            "chart":   "Graph",   # backward-compat alias
        }.get(figure_type, "Figure")

        header_lines = [
            f"Figure Type: {type_label}",
            f"Component: {part_number}",
            f"Page: {page}" if page else "",
        ]
        if caption:
            header_lines.insert(1, f"Caption: {caption}")

        header = "\n".join(l for l in header_lines if l)

        if vision_text:
            return f"{header}\n\n{vision_text}"
        elif caption:
            # Caption-only: repeat caption as Description so the embedding
            # captures both the structured header and the natural-language text.
            return f"{header}\n\nDescription: {caption}"
        else:
            return f"{header}\n\nFigure with no caption or vision analysis."
