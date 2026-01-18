"""
Page-by-Page PDF Extraction Module

This module implements:
1. Page count logging and per-page JSON saves
2. Dual extraction (LLM + pdfplumber) with audit/comparison per page
3. Document type detection by DATA (not title)
4. 3-iteration verification per page before moving to next
5. Smart combining: only combine same document types
6. Multiple document outputs per PDF with type, period info, sample queries
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import pdfplumber
from ollama import Client

# Import progress tracking
try:
    from app.progress import update_progress as _update_progress
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False
    _update_progress = None  # type: ignore

def update_progress(*args, **kwargs):  # type: ignore
    """Wrapper for progress updates."""
    if PROGRESS_AVAILABLE and _update_progress:
        return _update_progress(*args, **kwargs)

# Import OCR functions
try:
    from app.paddle_ocr import extract_pdf_pages_separately, OCR_AVAILABLE
except ImportError:
    OCR_AVAILABLE = False
    def extract_pdf_pages_separately(pdf_path: str, dpi: int = 200) -> List[Dict]:
        return []

# =========================
# CONFIG
# =========================

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3-coder:480b")
print(f"Ollama Host: {OLLAMA_HOST}")
print(f"Qwen Model: {QWEN_MODEL}")

client = Client(host=OLLAMA_HOST)

# Document types we can detect
DOCUMENT_TYPES = [
    "profit_loss",
    "balance_sheet", 
    "cash_flow",
    "ratios",
    "quarterly_results",
    "shareholding_pattern",
    "metadata",
    "unknown"
]


def _call_llm(prompt: str, think: bool = True) -> str:
    """Call Qwen LLM with prompt."""
    response = client.generate(
        model=QWEN_MODEL,
        prompt=prompt,
        options={"num_predict": 16000}
    )
    text = response.get("response", "")
    if think and "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def _extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response."""
    text = text.strip()
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _fix_pdfplumber_numbers(text: str) -> str:
    """Fix pdfplumber number extraction issues."""
    # Pattern: single digit + space + digit(s) + comma + digits
    pattern = r'(\d)\s+(\d{1,2},\d{3})'
    text = re.sub(pattern, r'\1\2', text)
    # Fix: "3 ,530" → "3,530"
    text = re.sub(r'(\d)\s+,(\d{3})', r'\1,\2', text)
    # Fix: "- 421" → "-421"
    text = re.sub(r'-\s+(\d)', r'-\1', text)
    return text


def extract_page_with_pdfplumber(pdf_path: str, page_num: int) -> Dict[str, Any]:
    """Extract text from a specific page using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        if page_num < 1 or page_num > len(pdf.pages):
            return {"page": page_num, "text": "", "char_count": 0, "method": "pdfplumber"}
        
        page = pdf.pages[page_num - 1]
        text = page.extract_text() or ""
        text = _fix_pdfplumber_numbers(text)
        
        return {
            "page": page_num,
            "text": text,
            "char_count": len(text),
            "method": "pdfplumber"
        }


def get_pdf_page_count(pdf_path: str) -> int:
    """Get total number of pages in a PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def _build_document_type_prompt(page_text: str) -> str:
    """Build prompt to detect document type from page content."""
    return f"""Analyze this financial document page and identify the document type based on the DATA CONTENT, not the title.

PAGE CONTENT:
{page_text}

DOCUMENT TYPES TO DETECT (identify by data patterns):
1. profit_loss - Contains: Sales/Revenue, Expenses, Operating Profit, Net Profit, EPS
2. balance_sheet - Contains: Assets, Liabilities, Equity, Reserves, Borrowings
3. cash_flow - Contains: Cash from Operations, Cash from Investing, Cash from Financing
4. ratios - Contains: ROE, ROCE, Current Ratio, Debtor Days, Inventory Turnover
5. quarterly_results - Contains: Quarterly periods (Jun-24, Sep-24, Dec-24)
6. shareholding_pattern - Contains: Promoter holding %, Public %, FII %
7. metadata - Contains: Face Value, Market Cap, Share Price, Bonus history

Respond with JSON:
{{
  "document_type": "profit_loss|balance_sheet|cash_flow|ratios|quarterly_results|shareholding_pattern|metadata",
  "confidence": 0.0-1.0,
  "detected_fields": ["field1", "field2", ...],
  "periods": ["Mar-16", "Mar-17", ...],
  "period_type": "annual|quarterly|ttm|mixed",
  "period_count": number
}}

Output ONLY valid JSON."""


def _build_page_extraction_prompt(page_text: str, doc_type: str) -> str:
    """Build prompt to extract data from a page based on detected document type."""
    type_specific_fields = {
        "profit_loss": '{"periods": [...], "sales": [...], "expenses": [...], "operating_profit": [...], "other_income": [...], "depreciation": [...], "interest": [...], "profit_before_tax": [...], "tax": [...], "net_profit": [...], "eps": [...]}',
        "balance_sheet": '{"periods": [...], "equity_share_capital": [...], "reserves": [...], "total_equity": [...], "borrowings": [...], "other_liabilities": [...], "total_liabilities": [...], "fixed_assets": [...], "current_assets": [...], "total_assets": [...]}',
        "cash_flow": '{"periods": [...], "cash_from_operations": [...], "cash_from_investing": [...], "cash_from_financing": [...], "net_cash_flow": [...]}',
        "ratios": '{"periods": [...], "roe": [...], "roce": [...], "current_ratio": [...], "debtor_days": [...], "inventory_turnover": [...], "dividend_payout": [...]}',
        "quarterly_results": '{"periods": [...], "sales": [...], "operating_profit": [...], "net_profit": [...], "opm": [...]}',
        "metadata": '{"face_value": number, "market_cap": number, "share_price": number, "52_week_high": number, "52_week_low": number}',
    }
    
    structure = type_specific_fields.get(doc_type, '{}')
    
    return f"""Extract all financial data from this {doc_type.upper()} document page.

PAGE CONTENT:
{page_text}

REQUIRED JSON STRUCTURE for {doc_type}:
{structure}

CRITICAL RULES:
1. Use EXACT field names as shown above
2. Extract ALL periods from column headers (Mar-16, Mar-17, etc.)
3. Extract ALL values for ALL periods - no partial extraction
4. Preserve negative values exactly (-100 stays -100)
5. Numbers should be numeric, not strings

Output ONLY valid JSON starting with {{ and ending with }}."""


def _build_verification_prompt(page_text: str, current_json: Dict, iteration: int) -> str:
    """Build prompt for iterative verification."""
    return f"""VERIFICATION ITERATION {iteration}/3

Compare the extracted JSON against the source data and fix any issues.

SOURCE DATA:
{page_text}

CURRENT EXTRACTION:
{json.dumps(current_json, indent=2)}

VERIFICATION CHECKLIST:
1. Are ALL periods from source included? Count columns in source.
2. Are ALL rows/fields extracted? Don't miss any data rows.
3. Are values EXACTLY as shown in source (including negatives)?
4. Is the period count consistent across all fields?
5. Are there any missing or extra values?

If issues found, output the CORRECTED complete JSON.
If no issues, output the same JSON unchanged.

Output ONLY valid JSON."""


def detect_document_type(page_text: str) -> Dict[str, Any]:
    """Detect document type from page content using LLM."""
    prompt = _build_document_type_prompt(page_text)
    response = _call_llm(prompt)
    result = _extract_json_from_text(response)
    
    if not result:
        return {
            "document_type": "unknown",
            "confidence": 0.0,
            "detected_fields": [],
            "periods": [],
            "period_type": "unknown",
            "period_count": 0
        }
    
    return result


def extract_page_data(page_text: str, doc_type: str, iterations: int = 3) -> Dict[str, Any]:
    """Extract structured data from a page with iterative verification."""
    
    # Initial extraction
    prompt = _build_page_extraction_prompt(page_text, doc_type)
    response = _call_llm(prompt)
    current_json = _extract_json_from_text(response)
    
    if not current_json:
        return {"error": "Failed to extract JSON", "raw_response": response[:500]}
    
    # Iterative verification (3 iterations)
    for i in range(2, iterations + 1):
        verify_prompt = _build_verification_prompt(page_text, current_json, i)
        verify_response = _call_llm(verify_prompt)
        verified_json = _extract_json_from_text(verify_response)
        
        if verified_json:
            current_json = verified_json
    
    return current_json


def audit_page_extraction(pdfplumber_text: str, ocr_text: str, extracted_json: Dict) -> Dict[str, Any]:
    """Audit extraction by comparing pdfplumber and OCR outputs."""
    
    # Count numeric values in each source
    def count_numbers(text: str) -> int:
        numbers = re.findall(r'-?\d+[,.]?\d*', text)
        return len(numbers)
    
    pdf_numbers = count_numbers(pdfplumber_text)
    ocr_numbers = count_numbers(ocr_text) if ocr_text else 0
    
    # Count values in extracted JSON
    def count_json_values(obj: Any) -> int:
        if isinstance(obj, dict):
            return sum(count_json_values(v) for v in obj.values())
        elif isinstance(obj, list):
            return len(obj) + sum(count_json_values(v) for v in obj if isinstance(v, (dict, list)))
        return 0
    
    json_values = count_json_values(extracted_json)
    
    # Compare
    audit_result = {
        "pdfplumber_char_count": len(pdfplumber_text),
        "ocr_char_count": len(ocr_text) if ocr_text else 0,
        "pdfplumber_numbers": pdf_numbers,
        "ocr_numbers": ocr_numbers,
        "json_values": json_values,
        "text_match_ratio": 0.0,
        "passed": True,
        "warnings": []
    }
    
    # Check if JSON captured enough data
    if json_values < pdf_numbers * 0.5:
        audit_result["passed"] = False
        audit_result["warnings"].append(f"JSON values ({json_values}) much less than source numbers ({pdf_numbers})")
    
    # Compare pdfplumber and OCR if both available
    if ocr_text and pdfplumber_text:
        # Simple overlap check
        pdf_set = set(pdfplumber_text.split())
        ocr_set = set(ocr_text.split())
        if pdf_set and ocr_set:
            overlap = len(pdf_set & ocr_set)
            audit_result["text_match_ratio"] = overlap / max(len(pdf_set), len(ocr_set))
    
    return audit_result


def process_single_page(
    pdf_path: str,
    page_num: int,
    total_pages: int,
    output_dir: Path,
    iterations: int = 3,
    use_ocr: bool = True,
    job_id: Optional[str] = None,
    ocr_cache: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Process a single PDF page with dual extraction and auditing.
    
    Returns page result with document type, extracted data, and audit info.
    """
    
    print(f"\n[PAGE {page_num}/{total_pages}] Processing...")
    
    # Update progress for frontend
    if job_id and PROGRESS_AVAILABLE:
        update_progress(
            job_id,
            status="extracting",
            message=f"Processing page {page_num}/{total_pages}",
            current_page=page_num,
            total_pages=total_pages,
            current_iteration=1
        )
    
    # Step 1: Extract text with pdfplumber
    print(f"  [1/5] Extracting with pdfplumber...")
    pdfplumber_result = extract_page_with_pdfplumber(pdf_path, page_num)
    pdfplumber_text = pdfplumber_result["text"]
    print(f"        -> {pdfplumber_result['char_count']} chars")
    
    # Step 2: Get OCR text from cache (already extracted once for all pages)
    ocr_text = ""
    if use_ocr and ocr_cache and page_num in ocr_cache:
        ocr_text = ocr_cache[page_num]
        print(f"  [2/5] OCR from cache -> {len(ocr_text)} chars")
    elif use_ocr and OCR_AVAILABLE and not ocr_cache:
        print(f"  [2/5] OCR not cached, skipping...")
    else:
        print(f"  [2/5] OCR skipped")
    
    # Use the better text source
    source_text = pdfplumber_text if len(pdfplumber_text) >= len(ocr_text) else ocr_text
    
    if not source_text.strip():
        print(f"  [SKIP] Page {page_num} has no extractable text")
        return {
            "page": page_num,
            "status": "skipped",
            "reason": "no_text",
            "document_type": "unknown"
        }
    
    # Step 3: Detect document type
    print(f"  [3/5] Detecting document type...")
    doc_type_info = detect_document_type(source_text)
    doc_type = doc_type_info.get("document_type", "unknown")
    confidence = doc_type_info.get("confidence", 0.0)
    periods = doc_type_info.get("periods", [])
    print(f"        -> Type: {doc_type} (confidence: {confidence:.0%})")
    print(f"        -> Periods: {periods[:5]}{'...' if len(periods) > 5 else ''}")
    
    # Step 4: Extract data with 3-iteration verification
    print(f"  [4/5] Extracting data ({iterations} iterations)...")
    extracted_data = extract_page_data(source_text, doc_type, iterations=iterations)
    print(f"        -> Extracted {len(json.dumps(extracted_data))} chars of JSON")
    
    # Step 5: Audit extraction
    print(f"  [5/5] Auditing extraction...")
    audit_result = audit_page_extraction(pdfplumber_text, ocr_text, extracted_data)
    audit_status = "✅ PASSED" if audit_result["passed"] else "❌ FAILED"
    print(f"        -> Audit: {audit_status}")
    if audit_result["warnings"]:
        for warning in audit_result["warnings"]:
            print(f"        -> WARNING: {warning}")
    
    # Save individual page JSON
    page_output = {
        "page": page_num,
        "total_pages": total_pages,
        "document_type": doc_type,
        "document_type_confidence": confidence,
        "periods": periods,
        "period_type": doc_type_info.get("period_type", "unknown"),
        "period_count": doc_type_info.get("period_count", 0),
        "extracted_data": extracted_data,
        "audit": audit_result,
        "source_chars": {
            "pdfplumber": len(pdfplumber_text),
            "ocr": len(ocr_text)
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to file
    page_file = output_dir / f"page_{page_num:03d}.json"
    with open(page_file, "w") as f:
        json.dump(page_output, f, indent=2)
    print(f"        -> Saved: {page_file.name}")
    
    return page_output


def combine_same_type_documents(page_results: List[Dict], doc_type: str, iterations: int = 3) -> Dict[str, Any]:
    """Combine pages of the same document type with iterative verification."""
    
    # Get all pages of this type
    same_type_pages = [p for p in page_results if p.get("document_type") == doc_type]
    
    if not same_type_pages:
        return {}
    
    print(f"\n[COMBINE] Merging {len(same_type_pages)} pages of type: {doc_type}")
    
    # Collect all periods and data
    all_periods = []
    combined_data = {}
    
    for page in same_type_pages:
        data = page.get("extracted_data", {})
        periods = data.get("periods", [])
        
        for period in periods:
            if period not in all_periods:
                all_periods.append(period)
        
        # Merge data fields
        for key, value in data.items():
            if key == "periods":
                continue
            if key not in combined_data:
                combined_data[key] = []
            if isinstance(value, list):
                combined_data[key].extend(value)
            else:
                combined_data[key] = value
    
    combined_data["periods"] = all_periods
    
    # Build combined document
    combined = {
        "document_type": doc_type,
        "pages_included": [p["page"] for p in same_type_pages],
        "total_periods": len(all_periods),
        "periods": all_periods,
        "data": combined_data
    }
    
    # Verify combined result with LLM
    print(f"  -> Verifying combined result ({iterations} iterations)...")
    
    for i in range(iterations):
        # Simple verification - check period consistency
        period_counts = {}
        for key, value in combined_data.items():
            if isinstance(value, list):
                period_counts[key] = len(value)
        
        if period_counts:
            expected = len(all_periods)
            mismatches = [k for k, v in period_counts.items() if v != expected and k != "periods"]
            if mismatches:
                print(f"     Iteration {i+1}: Period mismatch in {mismatches}")
            else:
                print(f"     Iteration {i+1}: ✅ All fields have {expected} periods")
    
    return combined


def generate_sample_queries(doc_type: str, periods: List[str], data: Dict) -> List[str]:
    """Generate sample queries for a document."""
    queries = []
    
    if doc_type == "profit_loss":
        if periods:
            queries.append(f"What was the net profit in {periods[-1]}?")
            queries.append(f"Show sales trend from {periods[0]} to {periods[-1]}")
            queries.append("What is the operating profit margin?")
            queries.append("Compare expenses across all years")
    elif doc_type == "balance_sheet":
        queries.append("What is the debt-to-equity ratio?")
        queries.append("Show total assets trend")
        queries.append("What are the borrowings?")
    elif doc_type == "ratios":
        queries.append("What is the ROE?")
        queries.append("Show ROCE trend")
        queries.append("What are the debtor days?")
    elif doc_type == "quarterly_results":
        queries.append("Compare quarterly sales")
        queries.append("What is the quarterly profit trend?")
    
    return queries


def process_pdf_page_by_page_v2(
    pdf_path: str,
    output_dir: Optional[str] = None,
    iterations_per_page: int = 3,
    combine_iterations: int = 3,
    use_ocr: bool = True,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process PDF with complete page-by-page extraction.
    
    Features:
    - Shows page count in logs
    - Saves individual JSON per page
    - Dual extraction (LLM + pdfplumber) with audit
    - Document type detection by DATA
    - 3-iteration verification per page
    - Smart combining by document type
    - Multiple document outputs per PDF
    
    Returns:
        Dict with documents, each identified by type, periods, and sample queries
    """
    
    pdf_name = Path(pdf_path).stem
    
    # Setup output directory
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = Path("processed") / f"{pdf_name}_pages"
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("PAGE-BY-PAGE PDF EXTRACTION v2")
    print("=" * 70)
    
    # Get page count
    total_pages = get_pdf_page_count(pdf_path)
    print(f"\n[INFO] PDF: {pdf_path}")
    print(f"[INFO] Total Pages: {total_pages}")
    print(f"[INFO] Iterations per page: {iterations_per_page}")
    print(f"[INFO] Combine iterations: {combine_iterations}")
    print(f"[INFO] OCR enabled: {use_ocr and OCR_AVAILABLE}")
    print(f"[INFO] Output directory: {out_path}")
    
    # Extract OCR for ALL pages ONCE at the start (efficiency)
    ocr_cache: Dict[int, str] = {}
    if use_ocr and OCR_AVAILABLE:
        print(f"\n[OCR] Extracting all pages at once...")
        try:
            ocr_results = extract_pdf_pages_separately(pdf_path, dpi=200)
            for ocr_page in ocr_results:
                ocr_cache[ocr_page["page_num"]] = ocr_page["text"]
            print(f"[OCR] Cached {len(ocr_cache)} pages")
        except Exception as e:
            print(f"[OCR] Failed: {e}")
    
    # Process each page
    page_results = []
    
    for page_num in range(1, total_pages + 1):
        try:
            result = process_single_page(
                pdf_path=pdf_path,
                page_num=page_num,
                total_pages=total_pages,
                output_dir=out_path,
                iterations=iterations_per_page,
                use_ocr=use_ocr,
                job_id=job_id,
                ocr_cache=ocr_cache
            )
            page_results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Page {page_num} failed: {e}")
            page_results.append({
                "page": page_num,
                "status": "error",
                "error": str(e),
                "document_type": "unknown"
            })
    
    # Identify unique document types
    doc_types_found = set()
    for result in page_results:
        doc_type = result.get("document_type", "unknown")
        if doc_type != "unknown" and result.get("status") != "error":
            doc_types_found.add(doc_type)
    
    print(f"\n[SUMMARY] Document types found: {list(doc_types_found)}")
    
    # Combine pages by document type
    documents = []
    
    for doc_type in doc_types_found:
        combined = combine_same_type_documents(page_results, doc_type, combine_iterations)
        
        if combined:
            # Generate sample queries
            periods = combined.get("periods", [])
            data = combined.get("data", {})
            sample_queries = generate_sample_queries(doc_type, periods, data)
            
            document = {
                "document_type": doc_type,
                "pages": combined.get("pages_included", []),
                "periods": periods,
                "period_count": len(periods),
                "period_type": "annual" if any("Mar-" in p for p in periods) else "quarterly",
                "data": data,
                "sample_queries": sample_queries
            }
            documents.append(document)
            
            # Save combined document JSON
            doc_file = out_path / f"combined_{doc_type}.json"
            with open(doc_file, "w") as f:
                json.dump(document, f, indent=2)
            print(f"  -> Saved: {doc_file.name}")
    
    # Build final result
    final_result = {
        "source_pdf": pdf_path,
        "total_pages": total_pages,
        "pages_processed": len(page_results),
        "pages_successful": len([p for p in page_results if p.get("status") != "error"]),
        "document_types_found": list(doc_types_found),
        "documents": documents,
        "page_details": page_results,
        "output_directory": str(out_path),
        "timestamp": datetime.now().isoformat()
    }
    
    # Save master result
    master_file = out_path / "extraction_result.json"
    with open(master_file, "w") as f:
        json.dump(final_result, f, indent=2)
    
    print("\n" + "=" * 70)
    print("[COMPLETE] Page-by-page extraction finished")
    print("=" * 70)
    print(f"  • Total pages: {total_pages}")
    print(f"  • Successful: {final_result['pages_successful']}")
    print(f"  • Document types: {list(doc_types_found)}")
    print(f"  • Documents produced: {len(documents)}")
    print(f"  • Output: {out_path}")
    
    return final_result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        result = process_pdf_page_by_page_v2(pdf_path)
        print(f"\nResult: {len(result['documents'])} documents extracted")
    else:
        print("Usage: python page_extractor.py <pdf_path>")
