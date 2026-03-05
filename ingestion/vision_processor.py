"""
ingestion/vision_processor.py
─────────────────────────────
Vision analysis for PDF figures using vikhyatk/moondream2 via HuggingFace.

Works in Google Colab (T4 GPU) without any sidecar service.

Model  : vikhyatk/moondream2  (~1.8 GB VRAM — fits alongside Qwen2.5-3B on T4)
Loading: Module-level singleton — loaded ONCE on the first VisionProcessor()
         instantiation, then reused for every subsequent figure. This means
         even though datasheet_chunker.py creates a new VisionProcessor(model=...)
         per figure, the model is never loaded more than once per process.
"""

from __future__ import annotations

import logging
import threading
from io import BytesIO
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ── Module-level singleton (shared across all VisionProcessor instances) ───────
_model     = None
_tokenizer = None
_load_lock = threading.Lock()

_MODEL_ID = "vikhyatk/moondream2"
_REVISION = "2025-01-09"   # pin for reproducibility


def _load_moondream() -> tuple:
    """Load moondream2 once into module globals; subsequent calls are a no-op."""
    global _model, _tokenizer

    with _load_lock:
        if _model is not None:
            return _tokenizer, _model

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed. Run: pip install transformers"
            ) from exc

        logger.info("Loading vision model %s  (revision=%s) …", _MODEL_ID, _REVISION)

        _tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_ID,
            revision=_REVISION,
            trust_remote_code=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            revision=_REVISION,
            trust_remote_code=True,
            device_map="auto",        # GPU when available, else CPU
        )
        _model.eval()
        logger.info("moondream2 ready ✓")

    return _tokenizer, _model


# ── VisionProcessor ────────────────────────────────────────────────────────────

class VisionProcessor:
    """
    Instantiate with:
        vision = VisionProcessor(model="moondream")   # existing call in chunker

    The `model` argument currently selects the backend; "moondream" maps to
    vikhyatk/moondream2 on HuggingFace. The architecture makes it easy to
    add other models later.
    """

    def __init__(self, model: str = "moondream") -> None:
        self.model_name = model

        # Eagerly load on first instantiation so the first figure call
        # doesn't incur hidden latency mid-pipeline.
        if self.model_name == "moondream":
            try:
                _load_moondream()
            except Exception as exc:
                # Non-fatal: if loading fails we fall back to caption-only mode.
                logger.warning(
                    "moondream2 failed to load — figures will use caption-only mode. "
                    "Error: %s", exc,
                )
        else:
            logger.warning(
                "Unknown vision model '%s'. Only 'moondream' is supported. "
                "Falling back to caption-only mode.", model,
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def describe_image(self, image_bytes: bytes, prompt: str) -> Optional[str]:
        """Run moondream2 on raw image bytes (PNG/JPEG) and return the answer.

        Parameters
        ----------
        image_bytes:
            Raw bytes from a PyMuPDF pixmap (PNG format).
        prompt:
            The question or instruction sent to the vision model.

        Returns
        -------
        str or None
            Model answer, or None if inference failed.
        """
        if self.model_name != "moondream" or _model is None:
            return None

        try:
            from PIL import Image

            tokenizer, model = _tokenizer, _model

            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

            # moondream2 API: encode image → answer question
            encoded = model.encode_image(pil_image)
            answer  = model.answer_question(encoded, prompt, tokenizer)

            return answer.strip() if answer else None

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
        """Crop a figure from a PDF page and describe it with moondream2.

        Parameters
        ----------
        pdf_path:
            Path to the source PDF.
        page_no:
            1-indexed page number (Docling provenance convention).
        bbox:
            Docling bbox dict — keys: l, t, r, b, coord_origin.
        is_graph:
            True → technical graph prompt; False → general diagram prompt.

        Returns
        -------
        str or None
            Vision model description, or None on failure.
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.warning("Vision: PDF not found at %s", pdf_path)
                return None

            with fitz.open(str(pdf_path)) as doc:
                page     = doc[page_no - 1]   # 1-indexed → 0-indexed
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

                # 2-pixel padding + 2× resolution for better model accuracy
                rect.x0 -= 2; rect.y0 -= 2
                rect.x1 += 2; rect.y1 += 2

                pix         = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
                image_bytes = pix.tobytes("png")

            prompt = (
                "This is a graph or chart from an electronics datasheet. "
                "State the x-axis label, y-axis label, their units, and the "
                "key trend or data points. Be precise and technical."
                if is_graph else
                "This is a diagram or image from an electronics datasheet. "
                "Describe any components, symbols, pin labels, dimensions, or "
                "circuit elements shown. Be precise and technical."
            )

            logger.info(
                "moondream2 — analysing figure page=%d is_graph=%s", page_no, is_graph
            )
            return self.describe_image(image_bytes, prompt)

        except Exception as exc:
            logger.error("Vision extraction failed: %s", exc, exc_info=True)
            return None
