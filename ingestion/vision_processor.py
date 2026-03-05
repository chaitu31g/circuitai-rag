"""
ingestion/vision_processor.py
─────────────────────────────
Vision analysis for PDF figures using vikhyatk/moondream2 via HuggingFace.

Replaces the Ollama-based integration so this works in Google Colab
(or any environment with a CUDA GPU) without a sidecar service.

Model  : vikhyatk/moondream2  (~1.8 GB, runs comfortably alongside Qwen2.5-3B on T4)
Loading: Lazy singleton — loaded only on first figure encountered, then reused.
Device : auto  (GPU if available, else CPU)
"""

from __future__ import annotations

import logging
import threading
from io import BytesIO
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ── Singleton state ────────────────────────────────────────────────────────────
_model     = None
_tokenizer = None
_load_lock = threading.Lock()

# Pin the revision for reproducibility — update when you want a newer moondream2
_MODEL_ID  = "vikhyatk/moondream2"
_REVISION  = "2025-01-09"


def _load_moondream() -> tuple:
    """Load moondream2 tokenizer + model once (thread-safe)."""
    global _model, _tokenizer

    with _load_lock:
        if _model is not None:
            return _tokenizer, _model

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers package missing — run: pip install transformers"
            ) from exc

        logger.info("Loading vision model: %s  revision=%s …", _MODEL_ID, _REVISION)

        _tokenizer = AutoTokenizer.from_pretrained(
            _MODEL_ID,
            revision=_REVISION,
            trust_remote_code=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID,
            revision=_REVISION,
            trust_remote_code=True,
            device_map="auto",       # GPU if CUDA available, else CPU
        )
        _model.eval()
        logger.info("Moondream2 loaded ✓  (device_map=auto)")

    return _tokenizer, _model


# ── Main processor class ───────────────────────────────────────────────────────

class VisionProcessor:
    """Crop a region from a PDF page and describe it with moondream2."""

    def describe_image(self, image_bytes: bytes, prompt: str) -> Optional[str]:
        """Run moondream2 on raw PNG/JPEG bytes and return the answer string.

        Parameters
        ----------
        image_bytes:
            Raw image bytes (PNG from PyMuPDF pixmap).
        prompt:
            The question / instruction for the vision model.

        Returns
        -------
        str or None
            The model's answer, or None if inference failed.
        """
        try:
            from PIL import Image  # Pillow must be available (PyMuPDF brings it)

            tokenizer, model = _load_moondream()

            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

            # Moondream2 API: encode image first, then answer question
            encoded = model.encode_image(pil_image)
            answer  = model.answer_question(
                encoded,
                prompt,
                tokenizer,
            )
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
        """Crop a bounding-box region from a PDF page and describe it.

        Parameters
        ----------
        pdf_path:
            Absolute path to the source PDF.
        page_no:
            1-indexed page number (as stored in Docling provenance).
        bbox:
            Docling bbox dict with keys l, t, r, b and optionally coord_origin.
        is_graph:
            True → use a more technical graph-analysis prompt.
            False → use a general diagram/component description prompt.

        Returns
        -------
        str or None
            Vision model description, or None if crop/inference failed.
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.warning("Vision: PDF not found at %s", pdf_path)
                return None

            with fitz.open(str(pdf_path)) as doc:
                page      = doc[page_no - 1]  # Docling pages are 1-indexed
                p_height  = page.rect.height

                l = bbox.get("l", 0)
                t = bbox.get("t", 0)
                r = bbox.get("r", 0)
                b = bbox.get("b", 0)

                # Docling uses BOTTOMLEFT origin; PyMuPDF uses TOPLEFT
                if bbox.get("coord_origin") == "BOTTOMLEFT":
                    rect = fitz.Rect(l, p_height - t, r, p_height - b)
                else:
                    rect = fitz.Rect(l, t, r, b)

                # Small padding to avoid cutting off edge pixels
                rect.x0 -= 2
                rect.y0 -= 2
                rect.x1 += 2
                rect.y1 += 2

                # Render at 2× scale for better OCR / model accuracy
                pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
                image_bytes = pix.tobytes("png")

            if is_graph:
                prompt = (
                    "This is a graph or chart from an electronics datasheet. "
                    "Describe the x-axis label and units, y-axis label and units, "
                    "and the key trends or data points shown. Be precise and technical."
                )
            else:
                prompt = (
                    "This is a diagram or image from an electronics datasheet. "
                    "Describe any components, symbols, pin labels, dimensions, "
                    "or circuit elements visible. Be technical and precise."
                )

            logger.info("Running moondream2 on figure (page %d, is_graph=%s)", page_no, is_graph)
            return self.describe_image(image_bytes, prompt)

        except Exception as exc:
            logger.error("Vision extraction failed: %s", exc, exc_info=True)
            return None
