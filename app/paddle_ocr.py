"""
OCR Integration for PDF Extraction

Provides OCR-based text and table extraction from PDFs,
especially useful for scanned documents and complex table layouts.

Supports:
- Tesseract OCR (pytesseract) - works with Python 3.14+
- PaddleOCR (fallback if available)
"""

import os
import json
import tempfile
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

# Lazy imports to avoid loading heavy libraries until needed
_ocr_engine = None
_pdf2image = None

# Check which OCR engine is available
TESSERACT_AVAILABLE = False
PADDLEOCR_AVAILABLE = False

try:
    import pytesseract
    # Configure Tesseract path for Windows
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break
    TESSERACT_AVAILABLE = True
except ImportError:
    pass

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    pass

# Use Tesseract if available (more compatible), else PaddleOCR
OCR_AVAILABLE = TESSERACT_AVAILABLE or PADDLEOCR_AVAILABLE


def _get_tesseract():
    """Get Tesseract OCR engine."""
    try:
        import pytesseract
        # Test if tesseract is installed
        pytesseract.get_tesseract_version()
        return pytesseract
    except Exception as e:
        raise ImportError(
            f"Tesseract OCR not available: {e}\n"
            "Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "Then: pip install pytesseract"
        )


def _get_paddleocr():
    """Lazy load PaddleOCR."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from paddleocr import PaddleOCR
            _ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False,
                use_gpu=False,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
            )
        except ImportError:
            raise ImportError(
                "PaddleOCR not installed. Run: pip install paddleocr paddlepaddle"
            )
    return _ocr_engine


def _get_pdf2image():
    """Lazy load pdf2image."""
    global _pdf2image
    if _pdf2image is None:
        try:
            from pdf2image import convert_from_path
            _pdf2image = convert_from_path
        except ImportError:
            raise ImportError(
                "pdf2image not installed. Run: pip install pdf2image\n"
                "Also requires poppler: https://github.com/oschwartz10612/poppler-windows/releases"
            )
    return _pdf2image


# Poppler path configuration for Windows
POPPLER_PATH = os.getenv("POPPLER_PATH", None)


def _find_poppler_path() -> Optional[str]:
    """Find poppler installation path on Windows."""
    if POPPLER_PATH:
        return POPPLER_PATH
    
    # Common installation locations on Windows
    common_paths = [
        r"C:\Program Files\poppler\Library\bin",
        r"C:\Program Files\poppler\bin",
        r"C:\poppler\Library\bin",
        r"C:\poppler\bin",
        os.path.expanduser(r"~\poppler\Library\bin"),
        os.path.expanduser(r"~\poppler\bin"),
    ]
    
    # Check for versioned directories
    for base in [r"C:\Program Files\poppler", r"C:\poppler"]:
        if os.path.exists(base):
            for item in os.listdir(base):
                if item.startswith("poppler-"):
                    versioned_bin = os.path.join(base, item, "Library", "bin")
                    if os.path.exists(versioned_bin):
                        common_paths.insert(0, versioned_bin)
                    versioned_bin2 = os.path.join(base, item, "bin")
                    if os.path.exists(versioned_bin2):
                        common_paths.insert(0, versioned_bin2)
    
    for path in common_paths:
        if os.path.exists(path) and os.path.isfile(os.path.join(path, "pdftoppm.exe")):
            return path
    
    return None


@dataclass
class OCRResult:
    """Result from OCR extraction."""
    text: str
    tables: List[Dict[str, Any]]
    raw_boxes: List[Dict[str, Any]]
    page_count: int
    confidence: float


def _extract_text_from_boxes(boxes: List) -> Tuple[str, List[Dict], float]:
    """
    Extract text and detect table structures from OCR boxes.
    
    Args:
        boxes: List of OCR detection results [(box_coords, (text, confidence)), ...]
    
    Returns:
        Tuple of (full_text, detected_tables, avg_confidence)
    """
    if not boxes:
        return "", [], 0.0
    
    # Sort boxes by vertical position (y-coordinate), then horizontal (x)
    sorted_boxes = []
    for box in boxes:
        if len(box) >= 2:
            coords = box[0]
            text_conf = box[1]
            if isinstance(text_conf, tuple) and len(text_conf) >= 2:
                text, conf = text_conf[0], text_conf[1]
                # Get top-left y coordinate for sorting
                y_pos = min(c[1] for c in coords)
                x_pos = min(c[0] for c in coords)
                sorted_boxes.append({
                    "coords": coords,
                    "text": text,
                    "confidence": conf,
                    "y": y_pos,
                    "x": x_pos,
                    "height": max(c[1] for c in coords) - y_pos,
                    "width": max(c[0] for c in coords) - x_pos,
                })
    
    if not sorted_boxes:
        return "", [], 0.0
    
    # Sort by y position, then x
    sorted_boxes.sort(key=lambda b: (b["y"], b["x"]))
    
    # Group boxes into lines (boxes with similar y coordinates)
    lines = []
    current_line = []
    line_y = sorted_boxes[0]["y"]
    y_threshold = 15  # Pixels threshold for same line
    
    for box in sorted_boxes:
        if abs(box["y"] - line_y) < y_threshold:
            current_line.append(box)
        else:
            if current_line:
                # Sort line by x position
                current_line.sort(key=lambda b: b["x"])
                lines.append(current_line)
            current_line = [box]
            line_y = box["y"]
    
    if current_line:
        current_line.sort(key=lambda b: b["x"])
        lines.append(current_line)
    
    # Build full text
    full_text_parts = []
    for line in lines:
        line_text = " ".join(b["text"] for b in line)
        full_text_parts.append(line_text)
    
    full_text = "\n".join(full_text_parts)
    
    # Calculate average confidence
    all_conf = [b["confidence"] for b in sorted_boxes]
    avg_confidence = sum(all_conf) / len(all_conf) if all_conf else 0.0
    
    # Detect tables by analyzing grid patterns
    tables = _detect_tables_from_boxes(sorted_boxes, lines)
    
    return full_text, tables, avg_confidence


def _detect_tables_from_boxes(boxes: List[Dict], lines: List[List[Dict]]) -> List[Dict]:
    """
    Detect table structures from OCR boxes.
    
    Looks for grid-like patterns where boxes align vertically and horizontally.
    """
    tables: List[Dict] = []
    
    if len(lines) < 2:
        return tables
    
    # Find lines that look like table rows (multiple aligned columns)
    table_candidate_lines = []
    
    for i, line in enumerate(lines):
        if len(line) >= 3:  # At least 3 columns suggests a table row
            # Check if boxes are evenly spaced (table-like)
            x_positions = [b["x"] for b in line]
            if len(x_positions) >= 3:
                # Calculate spacing variance
                spacings = [x_positions[j+1] - x_positions[j] for j in range(len(x_positions)-1)]
                if spacings:
                    avg_spacing = sum(spacings) / len(spacings)
                    if avg_spacing > 20:  # Minimum spacing for table
                        table_candidate_lines.append((i, line))
    
    # Group consecutive table-like lines into tables
    if table_candidate_lines:
        current_table_lines = []
        prev_idx = -2
        
        for idx, line in table_candidate_lines:
            if idx - prev_idx <= 2:  # Allow small gaps
                current_table_lines.append(line)
            else:
                if len(current_table_lines) >= 2:
                    tables.append(_build_table_from_lines(current_table_lines))
                current_table_lines = [line]
            prev_idx = idx
        
        if len(current_table_lines) >= 2:
            tables.append(_build_table_from_lines(current_table_lines))
    
    return tables


def _build_table_from_lines(lines: List[List[Dict]]) -> Dict[str, Any]:
    """Build a table structure from detected lines."""
    if not lines:
        return {"headers": [], "rows": []}
    
    # First line is header
    headers = [b["text"] for b in lines[0]]
    
    # Rest are data rows
    rows = []
    for line in lines[1:]:
        row = [b["text"] for b in line]
        # Pad row to match header length
        while len(row) < len(headers):
            row.append("")
        rows.append(row)
    
    return {
        "headers": headers,
        "rows": rows,
        "row_count": len(rows),
        "column_count": len(headers),
    }


def extract_pdf_with_ocr(
    pdf_path: str,
    dpi: int = 200,
    pages: Optional[List[int]] = None
) -> OCRResult:
    """
    Extract text and tables from PDF using PaddleOCR.
    
    Args:
        pdf_path: Path to PDF file
        dpi: DPI for PDF to image conversion (higher = better quality but slower)
        pages: Specific page numbers to process (1-indexed). None = all pages.
    
    Returns:
        OCRResult with extracted text, tables, and metadata
    """
    print(f"[PaddleOCR] Processing PDF: {pdf_path}")
    
    ocr = _get_paddleocr()
    convert_from_path = _get_pdf2image()
    
    # Convert PDF pages to images
    print(f"[PaddleOCR] Converting PDF to images (DPI={dpi})...")
    
    # Find poppler path for Windows
    poppler_path = _find_poppler_path()
    if poppler_path:
        print(f"[PaddleOCR] Using poppler from: {poppler_path}")
    
    try:
        if pages:
            # Convert specific pages
            images = []
            for page_num in pages:
                page_images = convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    first_page=page_num,
                    last_page=page_num,
                    poppler_path=poppler_path
                )
                images.extend(page_images)
        else:
            # Convert all pages
            images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    except Exception as e:
        if "poppler" in str(e).lower() or "pdftoppm" in str(e).lower():
            raise RuntimeError(
                f"Failed to convert PDF: {e}\n\n"
                "Poppler is required for PDF conversion. Install it:\n"
                "1. Download from: https://github.com/oschwartz10612/poppler-windows/releases\n"
                "2. Extract to C:\\Program Files\\poppler\\\n"
                "3. Add the bin folder to PATH, or set POPPLER_PATH environment variable"
            )
        raise RuntimeError(f"Failed to convert PDF to images: {e}")
    
    print(f"[PaddleOCR] Found {len(images)} page(s)")
    
    all_text_parts = []
    all_tables = []
    all_boxes = []
    total_confidence = 0.0
    
    # Process each page
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, image in enumerate(images):
            page_num = i + 1
            print(f"[PaddleOCR] Processing page {page_num}/{len(images)}...")
            
            # Save image temporarily
            img_path = os.path.join(tmpdir, f"page_{page_num}.png")
            image.save(img_path, "PNG")
            
            # Run OCR
            result = ocr.ocr(img_path, cls=True)
            
            if result and result[0]:
                page_text, page_tables, page_conf = _extract_text_from_boxes(result[0])
                
                all_text_parts.append(f"--- PAGE {page_num} ---\n{page_text}")
                all_tables.extend([{**t, "page": page_num} for t in page_tables])
                all_boxes.extend([{
                    "page": page_num,
                    "coords": b[0],
                    "text": b[1][0] if b[1] else "",
                    "confidence": b[1][1] if b[1] and len(b[1]) > 1 else 0.0
                } for b in result[0]])
                total_confidence += page_conf
            else:
                all_text_parts.append(f"--- PAGE {page_num} ---\n[No text detected]")
    
    full_text = "\n\n".join(all_text_parts)
    avg_confidence = total_confidence / len(images) if images else 0.0
    
    print(f"[PaddleOCR] Extraction complete: {len(full_text)} chars, {len(all_tables)} tables detected")
    
    return OCRResult(
        text=full_text,
        tables=all_tables,
        raw_boxes=all_boxes,
        page_count=len(images),
        confidence=avg_confidence,
    )


def extract_pdf_to_structured_json(
    pdf_path: str,
    dpi: int = 200
) -> Dict[str, Any]:
    """
    Extract PDF content to structured JSON format.
    
    Args:
        pdf_path: Path to PDF file
        dpi: DPI for conversion
    
    Returns:
        Structured JSON with text and tables
    """
    result = extract_pdf_with_ocr(pdf_path, dpi=dpi)
    
    return {
        "source_file": os.path.basename(pdf_path),
        "extraction_method": "paddleocr",
        "page_count": result.page_count,
        "confidence": round(result.confidence, 3),
        "full_text": result.text,
        "tables": result.tables,
        "raw_box_count": len(result.raw_boxes),
    }


def extract_pdf_with_tesseract(pdf_path: str, dpi: int = 200) -> OCRResult:
    """
    Extract text from PDF using Tesseract OCR.
    
    Args:
        pdf_path: Path to PDF file
        dpi: DPI for PDF to image conversion
    
    Returns:
        OCRResult with extracted text
    """
    print(f"[Tesseract] Processing PDF: {pdf_path}")
    
    pytesseract = _get_tesseract()
    convert_from_path = _get_pdf2image()
    
    # Find poppler path for Windows
    poppler_path = _find_poppler_path()
    if poppler_path:
        print(f"[Tesseract] Using poppler from: {poppler_path}")
    
    print(f"[Tesseract] Converting PDF to images (DPI={dpi})...")
    
    try:
        images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    except Exception as e:
        if "poppler" in str(e).lower() or "pdftoppm" in str(e).lower():
            raise RuntimeError(
                f"Failed to convert PDF: {e}\n\n"
                "Poppler is required. Install from:\n"
                "https://github.com/oschwartz10612/poppler-windows/releases"
            )
        raise RuntimeError(f"Failed to convert PDF: {e}")
    
    print(f"[Tesseract] Found {len(images)} page(s)")
    
    all_text_parts = []
    
    for i, image in enumerate(images):
        page_num = i + 1
        print(f"[Tesseract] Processing page {page_num}/{len(images)}...")
        
        # Run Tesseract OCR
        text = pytesseract.image_to_string(image)
        all_text_parts.append(f"--- PAGE {page_num} ---\n{text}")
    
    full_text = "\n\n".join(all_text_parts)
    
    print(f"[Tesseract] Extraction complete: {len(full_text)} chars")
    
    return OCRResult(
        text=full_text,
        tables=[],
        raw_boxes=[],
        page_count=len(images),
        confidence=0.9,  # Tesseract doesn't provide confidence per-page easily
    )


def get_ocr_text_for_llm(pdf_path: str, dpi: int = 200) -> str:
    """
    Get OCR-extracted text formatted for LLM processing.
    Uses Tesseract if available (Python 3.14 compatible), else PaddleOCR.
    
    Args:
        pdf_path: Path to PDF file
        dpi: DPI for conversion
    
    Returns:
        Text string suitable for LLM input
    """
    # Use Tesseract if available (more compatible with Python 3.14)
    if TESSERACT_AVAILABLE:
        result = extract_pdf_with_tesseract(pdf_path, dpi=dpi)
    elif PADDLEOCR_AVAILABLE:
        result = extract_pdf_with_ocr(pdf_path, dpi=dpi)
    else:
        raise ImportError(
            "No OCR engine available. Install one of:\n"
            "- Tesseract: pip install pytesseract (+ install Tesseract binary)\n"
            "- PaddleOCR: pip install paddleocr paddlepaddle"
        )
    
    # Format text with table markers
    output_parts = [result.text]
    
    if result.tables:
        output_parts.append("\n\n=== DETECTED TABLES ===\n")
        for i, table in enumerate(result.tables, 1):
            output_parts.append(f"\n--- Table {i} (Page {table.get('page', '?')}) ---")
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            
            if headers:
                output_parts.append(" | ".join(headers))
                output_parts.append("-" * (len(" | ".join(headers))))
            
            for row in rows:
                output_parts.append(" | ".join(str(c) for c in row))
    
    return "\n".join(output_parts)


# =========================
# HYBRID EXTRACTION
# =========================

def hybrid_pdf_extraction(
    pdf_path: str,
    use_pdfplumber: bool = True,
    use_paddleocr: bool = True,
    ocr_dpi: int = 200
) -> Dict[str, Any]:
    """
    Hybrid extraction using both pdfplumber (text-based) and PaddleOCR.
    
    Combines results for maximum coverage:
    - pdfplumber: Fast, accurate for text-based PDFs
    - PaddleOCR: Better for scanned docs, complex tables, images
    
    Args:
        pdf_path: Path to PDF file
        use_pdfplumber: Whether to use pdfplumber extraction
        use_paddleocr: Whether to use PaddleOCR extraction
        ocr_dpi: DPI for OCR conversion
    
    Returns:
        Combined extraction results
    """
    result: Dict[str, Any] = {
        "source_file": os.path.basename(pdf_path),
        "extraction_methods": [],
        "pdfplumber": None,
        "paddleocr": None,
        "combined_text": "",
    }
    
    # pdfplumber extraction
    if use_pdfplumber:
        try:
            import pdfplumber
            print("[Hybrid] Running pdfplumber extraction...")
            
            parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    parts.append(f"--- PAGE {i+1} ---\n{text}")
                    
                    # Try to extract tables
                    tables = page.extract_tables()
                    if tables:
                        for j, table in enumerate(tables):
                            parts.append(f"\n[Table {j+1}]")
                            for row in table:
                                parts.append(" | ".join(str(c) if c else "" for c in row))
            
            pdfplumber_text = "\n\n".join(parts)
            result["pdfplumber"] = {
                "text": pdfplumber_text,
                "char_count": len(pdfplumber_text),
            }
            result["extraction_methods"].append("pdfplumber")
            print(f"[Hybrid] pdfplumber: {len(pdfplumber_text)} chars")
        except Exception as e:
            print(f"[Hybrid] pdfplumber failed: {e}")
            result["pdfplumber"] = {"error": str(e)}
    
    # PaddleOCR extraction
    if use_paddleocr:
        try:
            print("[Hybrid] Running PaddleOCR extraction...")
            ocr_result = extract_pdf_with_ocr(pdf_path, dpi=ocr_dpi)
            result["paddleocr"] = {
                "text": ocr_result.text,
                "char_count": len(ocr_result.text),
                "tables": ocr_result.tables,
                "confidence": ocr_result.confidence,
            }
            result["extraction_methods"].append("paddleocr")
            print(f"[Hybrid] PaddleOCR: {len(ocr_result.text)} chars, {len(ocr_result.tables)} tables")
        except Exception as e:
            print(f"[Hybrid] PaddleOCR failed: {e}")
            result["paddleocr"] = {"error": str(e)}
    
    # Combine results - prefer longer text, merge unique content
    texts = []
    if result["pdfplumber"] and "text" in result["pdfplumber"]:
        texts.append(("pdfplumber", result["pdfplumber"]["text"]))
    if result["paddleocr"] and "text" in result["paddleocr"]:
        texts.append(("paddleocr", result["paddleocr"]["text"]))
    
    if texts:
        # Use the longer extraction as primary
        texts.sort(key=lambda t: len(t[1]), reverse=True)
        result["combined_text"] = texts[0][1]
        result["primary_method"] = texts[0][0]
    
    return result


# =========================
# CLI ENTRY POINT
# =========================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.paddle_ocr <pdf_path> [--dpi N] [--hybrid]")
        print("  pdf_path: Path to PDF file")
        print("  --dpi N: DPI for OCR (default: 200)")
        print("  --hybrid: Use hybrid extraction (pdfplumber + PaddleOCR)")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    dpi = 200
    hybrid = "--hybrid" in sys.argv
    
    if "--dpi" in sys.argv:
        idx = sys.argv.index("--dpi")
        if idx + 1 < len(sys.argv):
            dpi = int(sys.argv[idx + 1])
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    if hybrid:
        result = hybrid_pdf_extraction(pdf_path, ocr_dpi=dpi)
        print("\n" + "=" * 60)
        print("HYBRID EXTRACTION RESULT")
        print("=" * 60)
        print(f"Methods: {', '.join(result['extraction_methods'])}")
        print(f"Primary: {result.get('primary_method', 'none')}")
        print(f"Combined text length: {len(result['combined_text'])} chars")
    else:
        result = extract_pdf_to_structured_json(pdf_path, dpi=dpi)
        print("\n" + "=" * 60)
        print("PADDLEOCR EXTRACTION RESULT")
        print("=" * 60)
        print(f"Pages: {result['page_count']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Text length: {len(result['full_text'])} chars")
        print(f"Tables found: {len(result['tables'])}")
    
    # Save result
    output_path = os.path.splitext(pdf_path)[0] + "_ocr_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResult saved to: {output_path}")
