"""
ingestion/vision_processor.py
─────────────────────────────
Vision analysis for PDF figures using Qwen/Qwen2-VL-2B-Instruct via HuggingFace.

Replaces Moondream2. Qwen2-VL is significantly better at reading technical
graphs, axis labels, trends, and engineering diagrams from datasheets.

Model  : Qwen/Qwen2-VL-2B-Instruct (~5 GB, fits on T4 GPU with float16)
Loading: Module-level singleton — loaded ONCE on the first VisionProcessor()
         instantiation, shared across every figure in the pipeline.
Device : device_map="auto" (GPU layers auto-placed by Accelerate)
"""

from __future__ import annotations

import logging
import threading
from io import BytesIO
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ── Module-level singleton ─────────────────────────────────────────────────────
_model     = None
_processor = None
_device    = None
_load_lock = threading.Lock()

_DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def _load_qwen2vl(model_id: str = _DEFAULT_MODEL_ID) -> None:
    """Load Qwen2-VL processor + model once; subsequent calls are a no-op."""
    global _model, _processor, _device

    with _load_lock:
        if _model is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
        except ImportError as exc:
            raise RuntimeError(
                "Required packages missing. Run:\n"
                "  pip install transformers>=4.45 accelerate pillow"
            ) from exc

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading Qwen2-VL vision model: %s  (device=%s) …", model_id, _device
        )

        _processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        _model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",        # Qwen2-VL supports device_map natively
        )
        _model.eval()
        logger.info("Vision model loaded successfully ✓  (device_map=auto)")


# ── Prompt templates ───────────────────────────────────────────────────────────

_GRAPH_PROMPT = """\
You are analyzing a graph from an electronics component datasheet.

Explain:
• what the x-axis represents (variable name and units)
• what the y-axis represents (variable name and units)
• what relationship the graph shows between these variables
• the trend of the curve

Provide a concise technical explanation suitable for an engineer."""

_FIGURE_PROMPT = (
    "This image is from an electronics datasheet. "
    "Describe the diagram, component drawing, or table "
    "and summarize the important information shown."
)


# ── VisionProcessor ────────────────────────────────────────────────────────────

class VisionProcessor:
    """
    Drop-in replacement for the old Moondream-based processor.

    Usage (unchanged from datasheet_chunker.py):
        vision = VisionProcessor(model="qwen2-vl")
        desc   = vision.extract_and_describe(pdf_path, page, bbox, is_graph=True)
    """

    def __init__(self, model: str = "qwen2-vl") -> None:
        self.model_name = model
        model_id = _DEFAULT_MODEL_ID   # future: map other names here

        try:
            _load_qwen2vl(model_id)
        except Exception as exc:
            logger.warning(
                "Qwen2-VL failed to load — figures will use caption-only mode. "
                "Error: %s", exc,
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def describe_image(
        self,
        image_bytes: bytes,
        prompt: str = "",           # kept for backward-compat; overridden by is_graph
        is_graph: bool = False,
    ) -> Optional[str]:
        """Run Qwen2-VL on raw PNG image bytes and return a text description.

        Parameters
        ----------
        image_bytes : Raw bytes from a PyMuPDF pixmap (PNG).
        prompt      : Legacy/fallback prompt (ignored when is_graph determines template).
        is_graph    : True → use the detailed graph analysis prompt.

        Returns
        -------
        str or None
        """
        if _model is None or _processor is None:
            return None

        try:
            import torch
            from PIL import Image

            pil_image   = Image.open(BytesIO(image_bytes)).convert("RGB")
            text_prompt = _GRAPH_PROMPT if is_graph else _FIGURE_PROMPT

            # Build chat message with image + text
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text",  "text":  text_prompt},
                    ],
                }
            ]

            # Tokenise using the model's native chat template
            text = _processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = _processor(
                text=[text],
                images=[pil_image],
                padding=True,
                return_tensors="pt",
            ).to(_device)

            with torch.no_grad():
                generated_ids = _model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,        # deterministic — better for specs
                )

            # Strip the echoed input tokens
            trimmed = [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            result = _processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return result[0].strip() if result else None

        except Exception as exc:
            logger.error("Vision inference failed: %s", exc, exc_info=True)
            return None

    def extract_and_describe(
        self,
        pdf_path: Path,
        page_no: int,
        bbox: dict,
        is_graph: bool = False,
    ) -> Optional[str]:
        """Crop a figure region from a PDF page and describe it with Qwen2-VL.

        Parameters
        ----------
        pdf_path : Absolute path to the source PDF.
        page_no  : 1-indexed page number (Docling provenance convention).
        bbox     : Docling bbox dict — keys: l, t, r, b, coord_origin.
        is_graph : True → graph analysis prompt; False → diagram prompt.
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.warning("Vision: PDF not found at %s", pdf_path)
                return None

            with fitz.open(str(pdf_path)) as doc:
                page     = doc[page_no - 1]   # Docling 1-indexed → PyMuPDF 0-indexed
                p_height = page.rect.height

                l = bbox.get("l", 0)
                t = bbox.get("t", 0)
                r = bbox.get("r", 0)
                b = bbox.get("b", 0)

                # Docling BOTTOMLEFT origin → PyMuPDF TOPLEFT origin
                if bbox.get("coord_origin") == "BOTTOMLEFT":
                    rect = fitz.Rect(l, p_height - t, r, p_height - b)
                else:
                    rect = fitz.Rect(l, t, r, b)

                # 2-pixel padding + 2× super-sampling for model clarity
                rect.x0 -= 2; rect.y0 -= 2
                rect.x1 += 2; rect.y1 += 2

                pix         = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
                image_bytes = pix.tobytes("png")

            logger.info(
                "Analyzing figure page=%s is_graph=%s", page_no, is_graph
            )
            return self.describe_image(image_bytes, is_graph=is_graph)

        except Exception as exc:
            logger.error("Vision extraction failed: %s", exc, exc_info=True)
            return None
