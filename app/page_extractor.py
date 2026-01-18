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
    """Build prompt to detect MULTIPLE document types from page content."""
    return f"""Analyze this financial document page and identify ALL document types based on the DATA CONTENT.

CRITICAL: A single page may contain MULTIPLE distinct documents with DIFFERENT periods!
For example:
- METADATA section (company info, shares, face value) 
- ANNUAL PROFIT & LOSS (Mar-16, Mar-17, Mar-18, Mar-19)
- QUARTERLY RESULTS (Jun-23, Sep-23, Dec-23, Mar-24)

These are 3 SEPARATE documents with different period types - extract them separately!

PAGE CONTENT:
{page_text}

DOCUMENT TYPES TO DETECT (identify by data patterns):
1. profit_loss - Contains: Sales/Revenue, Expenses, Operating Profit, Net Profit, EPS (ANNUAL periods like Mar-XX)
2. quarterly_results - Contains: Quarterly periods (Jun-XX, Sep-XX, Dec-XX) with Sales, Profit
3. balance_sheet - Contains: Assets, Liabilities, Equity, Reserves, Borrowings
4. cash_flow - Contains: Cash from Operations, Cash from Investing, Cash from Financing
5. ratios - Contains: ROE, ROCE, Current Ratio, Debtor Days, Inventory Turnover
6. shareholding_pattern - Contains: Promoter holding %, Public %, FII %
7. metadata - Contains: Face Value, Market Cap, Share Price, Number of Shares

Respond with JSON containing an ARRAY of ALL document types found:
{{
  "documents": [
    {{
      "document_type": "metadata",
      "confidence": 0.95,
      "detected_fields": ["face_value", "market_cap", "shares"],
      "periods": [],
      "period_type": "none",
      "period_count": 0
    }},
    {{
      "document_type": "profit_loss",
      "confidence": 0.95,
      "detected_fields": ["sales", "expenses", "net_profit"],
      "periods": ["Mar-16", "Mar-17", "Mar-18", "Mar-19"],
      "period_type": "annual",
      "period_count": 4
    }},
    {{
      "document_type": "quarterly_results",
      "confidence": 0.95,
      "detected_fields": ["sales", "operating_profit", "net_profit"],
      "periods": ["Jun-23", "Sep-23", "Dec-23", "Mar-24"],
      "period_type": "quarterly",
      "period_count": 4
    }}
  ]
}}

RULES:
1. Separate documents by their PERIOD TYPE (annual vs quarterly vs none)
2. Never mix annual periods (Mar-16, Mar-17) with quarterly periods (Jun-23, Sep-23)
3. Metadata has no periods
4. Each distinct data section = separate document

Output ONLY valid JSON."""


def _build_page_extraction_prompt(page_text: str, doc_type: str, target_periods: List[str] = None) -> str:
    """Build prompt to extract data from a page based on detected document type and specific periods."""
    type_specific_fields = {
        "profit_loss": '{"periods": [...], "sales": [...], "expenses": [...], "operating_profit": [...], "other_income": [...], "depreciation": [...], "interest": [...], "profit_before_tax": [...], "tax": [...], "net_profit": [...], "eps": [...], "dividend_amount": [...]}',
        "balance_sheet": '{"periods": [...], "equity_share_capital": [...], "reserves": [...], "total_equity": [...], "borrowings": [...], "other_liabilities": [...], "total_liabilities": [...], "fixed_assets": [...], "current_assets": [...], "total_assets": [...]}',
        "cash_flow": '{"periods": [...], "cash_from_operations": [...], "cash_from_investing": [...], "cash_from_financing": [...], "net_cash_flow": [...]}',
        "ratios": '{"periods": [...], "roe": [...], "roce": [...], "current_ratio": [...], "debtor_days": [...], "inventory_turnover": [...], "dividend_payout": [...]}',
        "quarterly_results": '{"periods": [...], "sales": [...], "expenses": [...], "operating_profit": [...], "other_income": [...], "depreciation": [...], "interest": [...], "profit_before_tax": [...], "tax": [...], "net_profit": [...], "opm": [...]}',
        "metadata": '{"company_name": "...", "face_value": number, "market_cap": number, "share_price": number, "num_shares": number}',
    }
    
    structure = type_specific_fields.get(doc_type, '{}')
    
    period_instruction = ""
    if target_periods:
        period_instruction = f"""
TARGET PERIODS TO EXTRACT: {target_periods}
ONLY extract data for these specific periods. Do NOT include data from other periods."""
    
    return f"""Extract financial data for {doc_type.upper()} from this page.
{period_instruction}

PAGE CONTENT:
{page_text}

REQUIRED JSON STRUCTURE for {doc_type}:
{structure}

CRITICAL RULES:
1. Use EXACT field names as shown above
2. Extract ONLY the specified periods (if provided) - DO NOT MIX different period types
3. Annual periods look like: Mar-16, Mar-17, Mar-18, Mar-19
4. Quarterly periods look like: Jun-23, Sep-23, Dec-23, Mar-24
5. NEVER combine annual and quarterly data in the same JSON
6. Preserve negative values exactly (-100 stays -100)
7. Numbers should be numeric, not strings
8. Extract ALL available fields for the target period type

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


def detect_document_types(page_text: str) -> List[Dict[str, Any]]:
    """Detect ALL document types from page content using LLM. Returns list of documents."""
    prompt = _build_document_type_prompt(page_text)
    response = _call_llm(prompt)
    result = _extract_json_from_text(response)
    
    if not result:
        return [{
            "document_type": "unknown",
            "confidence": 0.0,
            "detected_fields": [],
            "periods": [],
            "period_type": "unknown",
            "period_count": 0
        }]
    
    # Handle new multi-document format
    if "documents" in result and isinstance(result["documents"], list):
        return result["documents"]
    
    # Backward compatibility: single document format
    return [result]


def detect_document_type(page_text: str) -> Dict[str, Any]:
    """Detect document type from page content using LLM. Legacy wrapper."""
    docs = detect_document_types(page_text)
    return docs[0] if docs else {
        "document_type": "unknown",
        "confidence": 0.0,
        "detected_fields": [],
        "periods": [],
        "period_type": "unknown",
        "period_count": 0
    }


def extract_page_data(page_text: str, doc_type: str, iterations: int = 3, target_periods: List[str] = None) -> Dict[str, Any]:
    """Extract structured data from a page with iterative verification."""
    
    # Initial extraction
    prompt = _build_page_extraction_prompt(page_text, doc_type, target_periods)
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
    
    Returns page result with MULTIPLE documents if different types/periods detected.
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
    print(f"  [1/6] Extracting with pdfplumber...")
    pdfplumber_result = extract_page_with_pdfplumber(pdf_path, page_num)
    pdfplumber_text = pdfplumber_result["text"]
    print(f"        -> {pdfplumber_result['char_count']} chars")
    
    # Step 2: Get OCR text from cache (already extracted once for all pages)
    ocr_text = ""
    if use_ocr and ocr_cache and page_num in ocr_cache:
        ocr_text = ocr_cache[page_num]
        print(f"  [2/6] OCR from cache -> {len(ocr_text)} chars")
    elif use_ocr and OCR_AVAILABLE and not ocr_cache:
        print(f"  [2/6] OCR not cached, skipping...")
    else:
        print(f"  [2/6] OCR skipped")
    
    # Use the better text source
    source_text = pdfplumber_text if len(pdfplumber_text) >= len(ocr_text) else ocr_text
    
    if not source_text.strip():
        print(f"  [SKIP] Page {page_num} has no extractable text")
        return {
            "page": page_num,
            "status": "skipped",
            "reason": "no_text",
            "documents": []
        }
    
    # Step 3: Detect ALL document types on this page
    print(f"  [3/6] Detecting document types (multi-doc)...")
    detected_docs = detect_document_types(source_text)
    print(f"        -> Found {len(detected_docs)} document type(s)")
    for doc in detected_docs:
        doc_type = doc.get("document_type", "unknown")
        periods = doc.get("periods", [])
        period_type = doc.get("period_type", "unknown")
        print(f"           • {doc_type} ({period_type}): {periods[:4]}{'...' if len(periods) > 4 else ''}")
    
    # Step 4: Extract data for EACH document type separately
    print(f"  [4/6] Extracting data for each document ({iterations} iterations each)...")
    all_documents = []
    
    for doc_info in detected_docs:
        doc_type = doc_info.get("document_type", "unknown")
        periods = doc_info.get("periods", [])
        period_type = doc_info.get("period_type", "unknown")
        confidence = doc_info.get("confidence", 0.0)
        
        print(f"        -> Extracting {doc_type} ({period_type})...")
        extracted_data = extract_page_data(
            source_text, 
            doc_type, 
            iterations=iterations,
            target_periods=periods if periods else None
        )
        
        # Audit this specific extraction
        audit_result = audit_page_extraction(pdfplumber_text, ocr_text, extracted_data)
        
        doc_output = {
            "document_type": doc_type,
            "document_type_confidence": confidence,
            "periods": periods,
            "period_type": period_type,
            "period_count": len(periods),
            "extracted_data": extracted_data,
            "audit": audit_result
        }
        all_documents.append(doc_output)
        print(f"           -> {len(json.dumps(extracted_data))} chars of JSON")
    
    # Step 5: Overall audit
    print(f"  [5/6] Auditing all extractions...")
    all_passed = all(d["audit"]["passed"] for d in all_documents)
    audit_status = "✅ PASSED" if all_passed else "❌ PARTIAL"
    print(f"        -> Audit: {audit_status} ({len(all_documents)} documents)")
    
    # Step 6: Save individual page JSON with all documents
    page_output = {
        "page": page_num,
        "total_pages": total_pages,
        "documents_on_page": len(all_documents),
        "documents": all_documents,
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
    print(f"  [6/6] Saved: {page_file.name}")
    
    return page_output


def combine_same_type_documents(page_results: List[Dict], doc_type: str, period_type: str = None, iterations: int = 3) -> Dict[str, Any]:
    """Combine documents of the same type AND period_type with iterative verification.
    
    Now handles new multi-document structure where each page has a "documents" array.
    """
    
    # Collect all matching documents from all pages
    matching_docs = []
    pages_included = []
    
    for page_result in page_results:
        page_num = page_result.get("page", 0)
        
        # Handle new structure with "documents" array
        if "documents" in page_result:
            for doc in page_result["documents"]:
                if doc.get("document_type") == doc_type:
                    # Also filter by period_type if specified
                    if period_type is None or doc.get("period_type") == period_type:
                        matching_docs.append(doc)
                        if page_num not in pages_included:
                            pages_included.append(page_num)
        # Handle legacy structure with single document
        elif page_result.get("document_type") == doc_type:
            if period_type is None or page_result.get("period_type") == period_type:
                matching_docs.append(page_result)
                pages_included.append(page_num)
    
    if not matching_docs:
        return {}
    
    type_label = f"{doc_type}" + (f" ({period_type})" if period_type else "")
    print(f"\n[COMBINE] Merging {len(matching_docs)} documents of type: {type_label}")
    
    # Collect all periods and data
    all_periods = []
    combined_data = {}
    
    for doc in matching_docs:
        data = doc.get("extracted_data", {})
        periods = data.get("periods", [])
        
        for period in periods:
            if period not in all_periods:
                all_periods.append(period)
        
        # Merge data fields
        for key, value in data.items():
            if key == "periods":
                continue
            if key not in combined_data:
                # Initialize based on value type
                if isinstance(value, list):
                    combined_data[key] = []
                else:
                    combined_data[key] = value
                    continue
            # Extend if both are lists
            if isinstance(value, list) and isinstance(combined_data[key], list):
                combined_data[key].extend(value)
            elif isinstance(value, list):
                # Value is list but existing is not - convert to list
                combined_data[key] = [combined_data[key]] + value
            elif isinstance(combined_data[key], list):
                # Existing is list but value is not - append
                combined_data[key].append(value)
            else:
                # Both are scalars - keep existing or overwrite
                combined_data[key] = value
    
    combined_data["periods"] = all_periods
    
    # Build combined document
    combined = {
        "document_type": doc_type,
        "period_type": period_type or "mixed",
        "pages_included": pages_included,
        "total_periods": len(all_periods),
        "periods": all_periods,
        "data": combined_data
    }
    
    # Verify combined result
    print(f"  -> Verifying combined result ({iterations} iterations)...")
    
    for i in range(iterations):
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
    
    # Identify unique document type + period_type combinations
    doc_type_combos = set()
    for result in page_results:
        if result.get("status") == "error":
            continue
        # Handle new multi-document structure
        if "documents" in result:
            for doc in result["documents"]:
                doc_type = doc.get("document_type", "unknown")
                period_type = doc.get("period_type", "unknown")
                if doc_type != "unknown":
                    doc_type_combos.add((doc_type, period_type))
        # Handle legacy structure
        elif result.get("document_type", "unknown") != "unknown":
            doc_type = result.get("document_type")
            period_type = result.get("period_type", "unknown")
            doc_type_combos.add((doc_type, period_type))
    
    print(f"\n[SUMMARY] Document type + period combinations: {list(doc_type_combos)}")
    
    # Combine by document type AND period_type
    documents = []
    
    for doc_type, period_type in doc_type_combos:
        combined = combine_same_type_documents(page_results, doc_type, period_type, combine_iterations)
        
        if combined:
            # Generate sample queries
            periods = combined.get("periods", [])
            data = combined.get("data", {})
            sample_queries = generate_sample_queries(doc_type, periods, data)
            
            document = {
                "document_type": doc_type,
                "period_type": period_type,
                "pages": combined.get("pages_included", []),
                "periods": periods,
                "period_count": len(periods),
                "data": data,
                "sample_queries": sample_queries
            }
            documents.append(document)
            
            # Save combined document JSON with period_type in filename
            safe_period_type = period_type.replace("/", "_") if period_type else "unknown"
            doc_file = out_path / f"combined_{doc_type}_{safe_period_type}.json"
            with open(doc_file, "w") as f:
                json.dump(document, f, indent=2)
            print(f"  -> Saved: {doc_file.name}")
    
    # Build final result
    final_result = {
        "source_pdf": pdf_path,
        "total_pages": total_pages,
        "pages_processed": len(page_results),
        "pages_successful": len([p for p in page_results if p.get("status") != "error"]),
        "document_types_found": [dt for dt, pt in doc_type_combos],
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
    print(f"  • Document types: {[dt for dt, pt in doc_type_combos]}")
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
