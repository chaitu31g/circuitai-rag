from typing import List, Dict, Any
import pdfplumber
import logging

logger = logging.getLogger(__name__)

class PdfTable:
    """Wrapper class for pdfplumber table extraction to match Docling bboxes."""
    def extract(self, pdf_path: str) -> List[Dict[str, Any]]:
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    # Custom settings handle borderless columns and solid row lines efficiently
                    t_settings = {"vertical_strategy": "text", "horizontal_strategy": "lines"}
                    page_tables = page.find_tables(table_settings=t_settings)
                    
                    for pt in page_tables:
                        try:
                            # pt.bbox is (x0, top, x1, bottom)
                            bbox = pt.bbox
                            data = pt.extract()
                            if not data:
                                continue
                                
                            tables.append({
                                "page": page_num,
                                "bbox": bbox,
                                "data": data  # list of lists of strings
                            })
                        except Exception as inner_e:
                            logger.warning(f"Failed to extract table on page {page_num}: {inner_e}")
                            continue
        except Exception as e:
            logger.error(f"PdfTable extraction failed for {pdf_path}: {e}")
            raise
            
        return tables
