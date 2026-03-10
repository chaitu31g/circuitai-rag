"""
ingestion/vision/figure_analyzer.py
────────────────────────────────────────────────────────────────────────────
Dual-model vision analyzer for electronics datasheet figures.

Routing logic
─────────────
  • Charts / graphs (axis-bearing images)  →  google/deplot
    DePlot is purpose-built for linearising chart images into a
    structured key-value table, giving much richer axis/value extraction
    than a generic VLM.

  • Diagrams, circuit drawings, package drawings  →  Qwen/Qwen2-VL-2B-Instruct
    Qwen2-VL produces detailed natural-language descriptions of complex
    visual elements that DePlot is not trained to handle.

Singleton pattern
─────────────────
Both models are loaded exactly ONCE at process startup (thread-safe).
Every call to FigureAnalyzer.analyze() reuses those shared instances.
Never load models inside loops.

Output format
─────────────
Regardless of which model was used, analyze() returns a structured
string that becomes the chunk text stored in ChromaDB:

  Figure Type: Graph          ← or "Diagram"
  Caption: <caption text>
  Component: <part_number>
  Page: <n>

  X-axis: ...                 ← DePlot path
  Y-axis: ...
  Extracted Data:
    25°C → 3.0 A
    ...
  Trend: ...

  ── or ──

  Description:                ← Qwen2-VL path
    <detailed description>
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
_QWEN_MODEL_ID   = "Qwen/Qwen2-VL-2B-Instruct"

# ── Module-level singletons — one set per model ───────────────────────────────
_deplot_processor = None
_deplot_model     = None

_qwen_processor   = None
_qwen_model       = None

_device           = None        # shared: 'cuda' or 'cpu'
_load_lock        = threading.Lock()

# ── Chart-detection keyword set ───────────────────────────────────────────────
_CHART_CAPTION_KEYWORDS = {
    "graph", "curve", "plot", "vs", "versus", "characteristics",
    "temperature", "current", "voltage", "power", "efficiency",
    "dissipation", "switching", "capacitance", "impedance", "frequency",
    "drain", "gate", "source", "transfer", "output", "soa",
    "safe operating", "thermal", "resistance", "response", "bandwidth",
}


# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (called once each, thread-safe)
# ═════════════════════════════════════════════════════════════════════════════

def _load_all_models() -> None:
    """Load DePlot and Qwen2-VL singletons. No-op on subsequent calls."""
    global _deplot_processor, _deplot_model
    global _qwen_processor, _qwen_model
    global _device

    with _load_lock:
        if _deplot_model is not None and _qwen_model is not None:
            return   # already loaded

        try:
            import torch
            from transformers import (
                AutoProcessor,
                AutoModelForVision2Seq,
                Pix2StructForConditionalGeneration,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Required packages missing. Run:\n"
                "  pip install transformers>=4.45 accelerate pillow"
            ) from exc

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype   = torch.float16 if _device == "cuda" else torch.float32

        # ── Load DePlot ───────────────────────────────────────────────────────
        if _deplot_model is None:
            logger.info("Loading DePlot: %s  (device=%s) …", _DEPLOT_MODEL_ID, _device)
            try:
                _deplot_processor = AutoProcessor.from_pretrained(_DEPLOT_MODEL_ID)
                _deplot_model = Pix2StructForConditionalGeneration.from_pretrained(
                    _DEPLOT_MODEL_ID,
                    torch_dtype=dtype,
                )
                _deplot_model.to(_device)
                _deplot_model.eval()
                logger.info("DePlot loaded ✓  (device=%s)", _device)
            except Exception as exc:
                logger.warning(
                    "DePlot failed to load — charts will fall back to Qwen2-VL. Error: %s", exc
                )
                _deplot_processor = None
                _deplot_model     = None

        # ── Load Qwen2-VL ─────────────────────────────────────────────────────
        if _qwen_model is None:
            logger.info("Loading Qwen2-VL: %s  (device=%s) …", _QWEN_MODEL_ID, _device)
            try:
                _qwen_processor = AutoProcessor.from_pretrained(
                    _QWEN_MODEL_ID,
                    trust_remote_code=True,
                )
                _qwen_model = AutoModelForVision2Seq.from_pretrained(
                    _QWEN_MODEL_ID,
                    torch_dtype=dtype,
                    device_map="auto",
                )
                _qwen_model.eval()
                logger.info("Qwen2-VL loaded ✓  (device_map=auto)")
            except Exception as exc:
                logger.warning(
                    "Qwen2-VL failed to load — diagrams will skip vision. Error: %s", exc
                )
                _qwen_processor = None
                _qwen_model     = None


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE TYPE CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════

def classify_figure(image_bytes: bytes, caption: str = "") -> str:
    """Classify a figure as either 'chart' or 'diagram'.

    Parameters
    ----------
    image_bytes : Raw PNG bytes of the figure.
    caption     : Caption text extracted from the PDF (may be empty).

    Returns
    -------
    'chart'   — axis-bearing graph/plot; route to DePlot.
    'diagram' — circuit drawing, package outline, block diagram; route to Qwen2-VL.
    """
    # 1. Caption keyword check (fast, no image decoding)
    cap_lower = caption.lower()
    if any(kw in cap_lower for kw in _CHART_CAPTION_KEYWORDS):
        logger.debug("classify_figure → chart  (caption keyword match)")
        return "chart"

    # 2. Simple pixel heuristic: look for near-horizontal / near-vertical
    #    lines that suggest axes or grid lines (uses basic numpy, no CV2 needed).
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(BytesIO(image_bytes)).convert("L")   # grayscale

        # Resize to a manageable width for speed
        max_w = 400
        if img.width > max_w:
            scale = max_w / img.width
            img   = img.resize((max_w, int(img.height * scale)), Image.LANCZOS)

        arr    = np.array(img, dtype=np.float32)
        # Threshold: pixels < 128 → dark (potential ink / line)
        dark   = (arr < 128).astype(np.uint8)

        h, w   = dark.shape
        # Row sums: a long horizontal dark line → axis or grid
        row_sum = dark.sum(axis=1)           # shape (h,)
        # Col sums: a long vertical dark line → axis
        col_sum = dark.sum(axis=0)           # shape (w,)

        # Heuristic: if ≥3 rows span > 60% of width → strong horizontal structure
        long_h_lines = (row_sum > 0.60 * w).sum()
        # Heuristic: if ≥3 cols span > 60% of height → strong vertical structure
        long_v_lines = (col_sum > 0.60 * h).sum()

        if long_h_lines >= 3 or long_v_lines >= 3:
            logger.debug(
                "classify_figure → chart  (pixel heuristic: h_lines=%d, v_lines=%d)",
                long_h_lines, long_v_lines,
            )
            return "chart"

        # 3. Numeric density check: many digit-like pixels near the edges
        #    (axis tick labels) suggests a chart.
        # (Lightweight proxy: check corner quadrant darker pixel fraction)
        edge_rows  = max(1, h // 8)
        edge_cols  = max(1, w // 8)
        top_strip  = dark[:edge_rows, :].mean()
        bot_strip  = dark[h - edge_rows:, :].mean()
        left_strip = dark[:, :edge_cols].mean()
        rgt_strip  = dark[:, w - edge_cols:].mean()

        if (top_strip + bot_strip + left_strip + rgt_strip) > 0.40:
            logger.debug(
                "classify_figure → chart  (dense edge labels heuristic)"
            )
            return "chart"

    except Exception as exc:
        logger.debug("Pixel-heuristic failed (continuing): %s", exc)

    logger.debug("classify_figure → diagram  (no chart features detected)")
    return "diagram"


# ═════════════════════════════════════════════════════════════════════════════
# DEPLOT PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

_DEPLOT_PROMPT = "Generate the underlying data table of the figure below:"

def _run_deplot(image_bytes: bytes) -> Optional[str]:
    """Send image to DePlot and return linearised chart data as text.

    DePlot output is already structured (e.g. 'TITLE | x1 | x2 / y1 | v1 | v2'),
    which we reformat into a human-readable block for chunking.
    """
    if _deplot_model is None or _deplot_processor is None:
        return None

    try:
        import torch
        from PIL import Image

        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Resize: DePlot works well up to 1024px on the long side
        max_px = 1024
        ratio  = min(max_px / max(img.width, img.height), 1.0)
        if ratio < 1.0:
            img = img.resize(
                (int(img.width * ratio), int(img.height * ratio)),
                Image.LANCZOS,
            )

        inputs = _deplot_processor(
            images=img,
            text=_DEPLOT_PROMPT,
            return_tensors="pt",
        ).to(_device)

        with torch.no_grad():
            generated_ids = _deplot_model.generate(
                **inputs,
                max_new_tokens=512,
            )

        raw = _deplot_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        # Reformat DePlot's pipe-separated output into a readable structure.
        return _format_deplot_output(raw)

    except Exception as exc:
        logger.error("DePlot inference failed: %s", exc, exc_info=True)
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
        # Split rows on  ' / '  (DePlot separator)
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
    # Try to extract numeric values from each row cell
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
# QWEN2-VL PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

_QWEN_DIAGRAM_PROMPT = (
    "This image is from an electronics datasheet. "
    "Describe the diagram, labels, connections, and technical meaning in detail."
)

def _run_qwen2vl(image_bytes: bytes) -> Optional[str]:
    """Send image to Qwen2-VL and return a detailed text description."""
    if _qwen_model is None or _qwen_processor is None:
        return None

    try:
        import torch
        from PIL import Image

        img = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Resize to avoid OOM on smaller GPUs (cap at 1024px long side, 2× zoom)
        max_px = 1024
        ratio  = min(max_px / max(img.width, img.height), 1.0)
        if ratio < 1.0:
            img = img.resize(
                (int(img.width * ratio), int(img.height * ratio)),
                Image.LANCZOS,
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text":  _QWEN_DIAGRAM_PROMPT},
                ],
            }
        ]

        text   = _qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _qwen_processor(
            text=[text],
            images=[img],
            padding=True,
            return_tensors="pt",
        ).to(_device)

        with torch.no_grad():
            generated_ids = _qwen_model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
            )

        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        result = _qwen_processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return result[0].strip() if result else None

    except Exception as exc:
        logger.error("Qwen2-VL inference failed: %s", exc, exc_info=True)
        return None


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC INTERFACE — FigureAnalyzer
# ═════════════════════════════════════════════════════════════════════════════

class FigureAnalyzer:
    """Unified figure analysis with automatic chart/diagram routing.

    Usage (replaces VisionProcessor in datasheet_chunker.py):

        analyzer = FigureAnalyzer()
        description = analyzer.extract_and_describe(
            pdf_path, page_no, bbox,
            caption="Drain Current vs Temperature",
            part_number="BSS84P",
        )
    """

    def __init__(self) -> None:
        try:
            _load_all_models()
        except Exception as exc:
            logger.warning(
                "FigureAnalyzer: one or more vision models failed to load — "
                "affected figure types will use caption-only mode. Error: %s", exc,
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
        figure_type = classify_figure(image_bytes, caption)
        logger.info(
            "FigureAnalyzer: page=%s  type=%s  part=%s  caption=%r",
            page, figure_type, part_number, caption[:60],
        )

        vision_text: Optional[str] = None

        if figure_type == "chart":
            vision_text = _run_deplot(image_bytes)
            if vision_text is None:
                logger.warning(
                    "DePlot returned None for chart on page %s — falling back to Qwen2-VL", page
                )
                vision_text = _run_qwen2vl(image_bytes)
        else:
            vision_text = _run_qwen2vl(image_bytes)

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
            "chart":   "Graph",
            "diagram": "Diagram",
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
            body_label = "Extracted Data / Analysis:" if figure_type == "chart" else "Description:"
            return f"{header}\n\n{body_label}\n{vision_text}"
        elif caption:
            return f"{header}\n\nNo vision analysis available. Caption: {caption}"
        else:
            return f"{header}\n\nFigure with no caption or vision analysis."
