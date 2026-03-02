import base64
import logging
import json
import http.client
from pathlib import Path
from typing import Optional, Any
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

class VisionProcessor:
    """Processes images and graphs from PDFs using local vision LLMs."""

    def __init__(self, model: str = "moondream"):
        self.model = model
        self.host = "localhost"
        self.port = 11434

    def describe_image(self, image_bytes: bytes, prompt: str) -> Optional[str]:
        """Send image to Ollama for description."""
        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            
            body = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "images": [image_b64],
                "options": {
                    "temperature": 0.1,
                    "num_predict": 300
                }
            }).encode("utf-8")

            conn = http.client.HTTPConnection(self.host, self.port, timeout=180)
            conn.request("POST", "/api/generate", body, {"Content-Type": "application/json"})
            
            resp = conn.getresponse()
            if resp.status != 200:
                logger.error(f"Ollama vision request failed: {resp.status} {resp.reason}")
                return None
            
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response")
        except Exception as e:
            logger.error(f"Error calling vision model: {e}")
            return None

    def extract_and_describe(
        self, 
        pdf_path: Path, 
        page_no: int, 
        bbox: dict, 
        is_graph: bool = False
    ) -> Optional[str]:
        """Extract a crop from the PDF and describe it."""
        try:
            if not pdf_path.exists():
                return None

            doc = fitz.open(str(pdf_path))
            # Page numbers in docling are 1-indexed
            page = doc[page_no - 1]
            
            # Convert Docling bbox to PyMuPDF Rect
            # Docling might use different coord origins. 
            # In zener_onsemi.json, coord_origin is "BOTTOMLEFT"
            # PyMuPDF Rect is (x0, y0, x1, y1) where y is from top.
            
            p_width = page.rect.width
            p_height = page.rect.height
            
            l, t, r, b = bbox.get("l"), bbox.get("t"), bbox.get("r"), bbox.get("b")
            
            if bbox.get("coord_origin") == "BOTTOMLEFT":
                # Convert bottom-up to top-down
                rect = fitz.Rect(l, p_height - t, r, p_height - b)
            else:
                rect = fitz.Rect(l, t, r, b)

            # Ensure some padding (2px)
            rect.x0 -= 2
            rect.y0 -= 2
            rect.x1 += 2
            rect.y1 += 2
            
            # Generate pixmap
            pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2)) # Higher resolution
            image_bytes = pix.tobytes("png")
            doc.close()

            if is_graph:
                prompt = (
                    "Look at this graph/chart from an electronics datasheet. "
                    "Tell me exactly what is on the xAxis and yAxis, the units used, "
                    "and the main trend or key data points shown. "
                    "Be technical and precise."
                )
            else:
                prompt = (
                    "Describe this image or diagram from an electronics datasheet in detail. "
                    "Identify symbols, components, or mechanical dimensions shown."
                )

            return self.describe_image(image_bytes, prompt)
        except Exception as e:
            logger.error(f"Vision extraction failed: {e}")
            return None
