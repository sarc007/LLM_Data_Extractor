"""
Qwen3 480B Cloud Pipeline for Data Extraction

This module uses Ollama's Qwen3 480B cloud model for:
1. PDF/Excel/CSV data extraction with 5-iteration self-check
2. Document type classification
3. Period detection and analysis
4. Sample query generation
5. PaddleOCR integration for scanned PDFs
6. Data audit verification
"""

import json
import os
import re
import tempfile
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime

import pandas as pd
import pdfplumber
from ollama import Client

# Optional OCR import (lazy loaded) - supports Tesseract and PaddleOCR
PADDLEOCR_AVAILABLE = False
OCR_AVAILABLE = False
try:
    from app.paddle_ocr import (
        get_ocr_text_for_llm, 
        hybrid_pdf_extraction, 
        extract_pdf_pages_separately,
        OCR_AVAILABLE, 
        TESSERACT_AVAILABLE
    )
    PADDLEOCR_AVAILABLE = OCR_AVAILABLE  # For backward compatibility
except ImportError:
    pass

# Optional Audit import
AUDIT_AVAILABLE = False
try:
    from app.audit import audit_extraction, AuditResult
    AUDIT_AVAILABLE = True
except ImportError:
    pass

# =========================
# CONFIG
# =========================

# Ollama Cloud API configuration
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3-coder:480b")

def get_ollama_client():
    """Get Ollama client configured for cloud API."""
    if OLLAMA_API_KEY:
        return Client(
            host=OLLAMA_HOST,
            headers={'Authorization': 'Bearer ' + OLLAMA_API_KEY}
        )
    else:
        # Fallback to local if no API key
        return Client(host="http://localhost:11434")

print(f"Ollama Host: {OLLAMA_HOST}")
print(f"Qwen Model: {QWEN_MODEL}")


# =========================
# JSON UTILITIES
# =========================

def _extract_json_from_text(text: str) -> Any:
    """Extract JSON from LLM response, handling markdown fences and cleanup."""
    # Remove thinking tags if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    # Try to find fenced JSON block
    fenced = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    else:
        # Try generic code fence
        fenced = re.search(r"```(.*?)```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

    # Find JSON object boundaries
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first:last + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Cleanup common issues
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        text = re.sub(r"'", '"', text)  # Single to double quotes
        return json.loads(text)


# =========================
# FILE CONVERTERS
# =========================

def _detect_file_type(path: str) -> str:
    """Detect file type from extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext in [".xls", ".xlsx"]:
        return "excel"
    if ext == ".csv":
        return "csv"
    return "unknown"


def _load_pdf_text(path: str) -> str:
    """Extract text from PDF file."""
    parts = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            parts.append(f"--- PAGE {i+1} ---\n{text}")
    return "\n\n".join(parts)


def _load_excel_text(path: str) -> str:
    """Load Excel file and convert to text representation for LLM."""
    xl = pd.ExcelFile(path)
    parts = []
    
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        if df.empty:
            continue
        
        # Convert to markdown table for better LLM understanding
        try:
            markdown = df.fillna("").to_markdown(index=False)
            parts.append(f"### SHEET: {sheet_name} ###\n{markdown}")
        except ImportError:
            # Fallback to CSV if tabulate not available
            csv_text = df.fillna("").to_csv(index=False)
            parts.append(f"### SHEET: {sheet_name} ###\n{csv_text}")
    
    return "\n\n".join(parts)


def _load_excel_as_json(path: str) -> Dict[str, Any]:
    """
    Load Excel file directly into structured JSON.
    Reads ALL sheets and preserves data types.
    Intelligently finds data sections within sheets.
    """
    xl = pd.ExcelFile(path)
    result = {
        "source_file": os.path.basename(path),
        "sheets": {}
    }
    
    for sheet_name in xl.sheet_names:
        # Read without headers first to scan the sheet
        df_raw = xl.parse(sheet_name, header=None)
        if df_raw.empty:
            continue
        
        # Find the best header row - look for rows with date-like values or period patterns
        best_header_row = 0
        for row_idx in range(min(20, len(df_raw))):
            row_vals = df_raw.iloc[row_idx].tolist()
            # Check if this row looks like headers (has dates or period names)
            date_count = sum(1 for v in row_vals if _looks_like_period(v))
            non_null = sum(1 for v in row_vals if pd.notna(v))
            
            if date_count >= 3 or (non_null >= 5 and row_idx > 0):
                # Check if the row below has numeric data
                if row_idx + 1 < len(df_raw):
                    next_row = df_raw.iloc[row_idx + 1].tolist()
                    numeric_count = sum(1 for v in next_row if isinstance(v, (int, float)) and pd.notna(v))
                    if numeric_count >= 3:
                        best_header_row = row_idx
                        break
        
        # Re-read with the detected header row
        df = xl.parse(sheet_name, header=best_header_row)
        
        # Clean up column names
        new_columns = []
        for i, col in enumerate(df.columns):
            col_str = str(col)
            if col_str.startswith("Unnamed") or pd.isna(col):
                new_columns.append(f"Column_{i+1}")
            elif hasattr(col, 'strftime'):  # It's a datetime
                # Format dates as Mar-YY
                new_columns.append(col.strftime("%b-%y"))
            else:
                new_columns.append(str(col).strip())
        df.columns = new_columns
        
        # Drop rows that are completely empty
        df = df.dropna(how='all')
        
        # Convert each row to a dict
        rows = []
        for _, row in df.iterrows():
            row_dict = {}
            has_numeric = False
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    row_dict[str(col)] = None
                elif isinstance(val, (int, float)):
                    row_dict[str(col)] = round(val, 2) if isinstance(val, float) else val
                    has_numeric = True
                else:
                    row_dict[str(col)] = str(val).strip()
            
            # Only add rows that have numeric data (actual data rows)
            if has_numeric:
                rows.append(row_dict)
        
        if rows:
            result["sheets"][sheet_name] = {
                "columns": list(df.columns),
                "row_count": len(rows),
                "data": rows
            }
    
    return result


def _looks_like_period(val) -> bool:
    """Check if a value looks like a date/period header."""
    if pd.isna(val):
        return False
    if hasattr(val, 'strftime'):  # datetime object
        return True
    val_str = str(val).lower()
    # Check for common period patterns
    patterns = ['mar-', 'jun-', 'sep-', 'dec-', 'jan-', 'feb-', 'apr-', 'may-', 
                'jul-', 'aug-', 'oct-', 'nov-', 'q1', 'q2', 'q3', 'q4', 
                '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
    return any(p in val_str for p in patterns)


def _load_csv_as_json(path: str) -> Dict[str, Any]:
    """
    Load CSV file directly into structured JSON.
    """
    df = pd.read_csv(path)
    
    columns = df.columns.tolist()
    rows = []
    
    for _, row in df.iterrows():
        row_dict = {}
        for col in columns:
            val = row[col]
            if pd.isna(val):
                row_dict[str(col)] = None
            elif isinstance(val, (int, float)):
                row_dict[str(col)] = val
            else:
                row_dict[str(col)] = str(val)
        rows.append(row_dict)
    
    return {
        "source_file": os.path.basename(path),
        "sheets": {
            "data": {
                "columns": [str(c) for c in columns],
                "row_count": len(rows),
                "data": rows
            }
        }
    }


def _load_csv_text(path: str) -> str:
    """Load CSV file and convert to text representation."""
    df = pd.read_csv(path)
    try:
        return df.fillna("").to_markdown(index=False)
    except ImportError:
        return df.fillna("").to_csv(index=False)


def convert_to_pdf(input_path: str, output_dir: str = None) -> str:
    """
    Convert Excel or CSV file to PDF for processing.
    Returns path to the converted PDF.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        print("Installing reportlab for PDF conversion...")
        os.system("pip install reportlab")
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    
    file_type = _detect_file_type(input_path)
    
    if file_type == "pdf":
        return input_path  # Already PDF
    
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_converted.pdf")
    
    # Load data
    if file_type == "excel":
        xl = pd.ExcelFile(input_path)
        sheets_data = [(sheet, xl.parse(sheet)) for sheet in xl.sheet_names]
    elif file_type == "csv":
        df = pd.read_csv(input_path)
        sheets_data = [("Data", df)]
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    # Create PDF
    doc = SimpleDocTemplate(output_path, pagesize=landscape(letter))
    styles = getSampleStyleSheet()
    elements = []
    
    for sheet_name, df in sheets_data:
        if df.empty:
            continue
        
        # Add sheet title
        elements.append(Paragraph(f"<b>Sheet: {sheet_name}</b>", styles['Heading2']))
        elements.append(Spacer(1, 12))
        
        # Convert DataFrame to table data
        df = df.fillna("")
        table_data = [df.columns.tolist()] + df.values.tolist()
        
        # Truncate long text in cells
        table_data = [
            [str(cell)[:50] + "..." if len(str(cell)) > 50 else str(cell) for cell in row]
            for row in table_data
        ]
        
        # Create table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 24))
    
    doc.build(elements)
    print(f"  -> Converted to PDF: {output_path}")
    return output_path


# =========================
# LLM CALLER
# =========================

def _call_qwen(prompt: str, system_prompt: str = None, timeout: int = 600) -> str:
    """
    Call Qwen model via Ollama cloud API.
    Returns the raw response text.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    try:
        client = get_ollama_client()
        response = client.chat(
            model=QWEN_MODEL,
            messages=messages,
            options={"temperature": 0}
        )
        return response['message']['content']
    except Exception as e:
        raise RuntimeError(f"Failed to call Qwen: {e}")


# =========================
# EXTRACTION PROMPTS
# =========================

# User's exact prompt for extraction with 5-iteration self-check
USER_EXTRACTION_PROMPT = """You are a financial data analyst. Extract all data from the document in clean JSON format."""


def _build_extraction_prompt(doc_text: str) -> str:
    """Build the extraction prompt with strict data quality rules."""
    return f"""{USER_EXTRACTION_PROMPT}

DOCUMENT CONTENT:
{doc_text}

CRITICAL RULES - FOLLOW EXACTLY:

1. DATE RANGE - USE ONLY ACTUAL PERIODS FROM DOCUMENT:
   - Look at the COLUMN HEADERS to find actual periods (e.g., Mar-16, Mar-17)
   - The FIRST period in your JSON must match the FIRST column in the document
   - Do NOT invent periods like Mar-13, Mar-14 if they don't exist in source
   - If document shows Mar-16 to Mar-25, output Mar-16 to Mar-25 ONLY

2. SEPARATE DATA BY FREQUENCY - NEVER MIX:
   - Annual data (Mar-16, Mar-17, Mar-18): Put in main "profit_loss", "balance_sheet" sections
   - Quarterly data (Q1, Jun-24, Sep-24): Put in separate "quarterly_data" section
   - TTM/Trailing values: Put in separate "ttm_data" section
   - Best Case/Worst Case: Put in "scenario_data" - these are NOT historical periods

3. COMPLETE EXTRACTION - ALL VALUES:
   - If a row has 10 columns, extract ALL 10 values (not just first 3 or last 3)
   - Operating profit, depreciation, interest, PBT, tax, net profit, EPS - ALL must be complete
   - Each metric should have the SAME number of periods

4. TABLE BOUNDARIES:
   - Profit & Loss, Balance Sheet, Cash Flow, Ratios = SEPARATE sections
   - Do NOT merge data from different tables

5. DATA FORMAT:
   - "metric": [{{"period": "Mar-16", "value": 1234.56}}, {{"period": "Mar-17", "value": 2345.67}}]
   - Percentages as strings: "27%"
   - Empty cells as null

Output ONLY valid JSON. Start with {{ and end with }}."""


def _build_ocr_to_json_prompt(ocr_text: str) -> str:
    """Build prompt to convert OCR-extracted text to structured JSON.
    
    This is used when OCR has already extracted the data - LLM just structures it.
    """
    return f"""You are a financial data structuring expert. Convert this OCR-extracted text to clean JSON.

OCR-EXTRACTED TEXT:
{ocr_text}

CRITICAL RULES - READ CAREFULLY:

1. DETECT ACTUAL DATE RANGE FROM THE DATA:
   - Look at the column headers to find the ACTUAL periods (e.g., "Mar-16", "Mar-17", etc.)
   - ONLY use periods that ACTUALLY APPEAR in the OCR text
   - Do NOT invent periods like Mar-13, Mar-14 if they don't exist in the source

2. SEPARATE DATA BY FREQUENCY - DO NOT MIX:
   - "annual_data": For fiscal year data (Mar-16, Mar-17, Mar-18, etc.)
   - "quarterly_data": For quarterly data (Q1, Q2, Jun-24, Sep-24, etc.) - KEEP SEPARATE
   - "ttm_data": For Trailing Twelve Months / TTM values - KEEP SEPARATE
   - "scenario_data": For Best Case, Worst Case projections - KEEP SEPARATE, NOT historical

3. TABLE BOUNDARY DETECTION:
   - Each distinct table (Profit & Loss, Balance Sheet, Cash Flow, Ratios) should be a separate section
   - If you see a new table header, START A NEW SECTION
   - Do NOT merge data from different tables

4. DATA FORMAT:
   - "metric_name": [{{"period": "Mar-16", "value": 1234.56}}, {{"period": "Mar-17", "value": 2345.67}}]
   - Percentages as strings: "27%" or "12.13%"
   - Empty cells or "-" as null
   - Numbers without currency symbols

5. EXTRACT COMPLETE SERIES:
   - Extract ALL values for each metric row, not just first/last few
   - If a row has 10 year columns, extract all 10 values
   - Do NOT skip middle values

JSON STRUCTURE:
{{
    "company_name": "...",
    "source": "...",
    "profit_loss": {{
        "sales": [...],
        "expenses": [...],
        "operating_profit": [...],
        "other_income": [...],
        "depreciation": [...],
        "interest": [...],
        "profit_before_tax": [...],
        "tax": [...],
        "net_profit": [...],
        "eps": [...]
    }},
    "balance_sheet": {{...}},
    "cash_flow": {{...}},
    "ratios": {{...}},
    "quarterly_data": {{...}},
    "ttm_data": {{...}},
    "scenario_data": {{"best_case": ..., "worst_case": ...}}
}}

Output ONLY valid JSON. Start with {{ and end with }}."""


def _build_selfcheck_prompt(doc_text: str, current_json: str, iteration: int, total_iterations: int) -> str:
    """Build prompt for self-checking extracted data - using user's iteration approach."""
    return f"""You are a financial data analyst. Self-check iteration {iteration} of {total_iterations}.

ORIGINAL DOCUMENT:
{doc_text}

YOUR CURRENT EXTRACTION:
{current_json}

VERIFICATION CHECKLIST:

1. DATE RANGE ACCURACY:
   - Are the periods in JSON matching EXACTLY what's in the document?
   - Did you invent any periods (Mar-13, Mar-14) that don't exist in source? REMOVE THEM
   - The FIRST period should match the FIRST column header in the document

2. DATA FREQUENCY SEPARATION:
   - Annual data (Mar-16, Mar-17) should be in main sections
   - Quarterly data (Q1, Jun-24) should be in "quarterly_data" - NOT mixed with annual
   - TTM/Trailing values should be in "ttm_data" - NOT mixed with annual
   - Best/Worst Case should be in "scenario_data" - NOT treated as historical

3. COMPLETE EXTRACTION:
   - Each metric row should have ALL values (if 10 columns, extract 10 values)
   - Operating profit, other income, depreciation, interest, PBT, tax, net profit, EPS - all complete?
   - No artificial drops or jumps in series due to mixed frequencies?

4. TABLE BOUNDARIES:
   - Profit & Loss, Balance Sheet, Cash Flow, Ratios - each in separate sections?
   - No data from one table bleeding into another?

Fix any issues found. Output COMPLETE corrected JSON. Start with {{ and end with }}."""


# =========================
# ANALYSIS PROMPTS
# =========================

def _build_analysis_prompt(extracted_json: str) -> str:
    """Build prompt to analyze the extracted data for the 4 key points."""
    return f"""Analyze this extracted financial/business data and determine:

EXTRACTED DATA:
{extracted_json}

Based on the data above, provide analysis in this EXACT JSON format:

{{
    "data_type": {{
        "primary_type": "Identify the type: 'Sales Report', 'Profit and Loss Statement', 'Balance Sheet', 'Cash Flow Statement', 'Revenue Report', etc.",
        "confidence": 0.95,
        "reasoning": "Brief explanation"
    }},
    "number_of_periods": 10,
    "period_type": "annual/quarterly/monthly/weekly/daily",
    "periods_found": ["List each period found, e.g., 'Mar-16', 'Mar-17', etc. Do NOT count 'Trailing', 'Best Case', 'Worst Case' as periods"],
    "sample_queries": [
        "Get me past 3 years sales data",
        "Compare sales growth between Mar-22 and Mar-24",
        "What was the net profit trend over the last 5 years",
        "Show me the operating profit margin for all years",
        "Which year had the highest EPS",
        "Compare expenses vs revenue for Mar-23",
        "What is the average sales growth rate",
        "Show profit before tax for the last 3 years"
    ]
}}

IMPORTANT:
- For number_of_periods: Count ONLY actual fiscal periods (like Mar-16, Mar-17, etc.). Do NOT count 'Trailing', 'Best Case', 'Worst Case' as periods.
- For period_type: If periods are like "Mar-16", "Mar-17" etc. (annual fiscal years), the type is "annual"
- sample_queries should be realistic queries someone would ask to analyze this specific data

Output ONLY valid JSON. Start with {{ and end with }}."""


# =========================
# MAIN PIPELINE
# =========================

def structure_single_page(page_text: str, page_num: int, total_pages: int, iterations: int = 5, job_id: str = None) -> Dict[str, Any]:
    """
    Structure a single page's OCR text to JSON with iterative verification.
    
    Args:
        page_text: OCR-extracted text from one page
        page_num: Current page number
        total_pages: Total number of pages
        iterations: Number of verification iterations (default 5)
    
    Returns:
        Structured JSON for this page
    """
    print(f"\n{'='*60}")
    print(f"[PAGE {page_num}/{total_pages}] Processing...")
    print(f"{'='*60}")
    
    # Import progress tracker if job_id provided
    if job_id:
        from app.progress import update_progress
        update_progress(job_id, current_page=page_num, current_iteration=0, 
                       message=f"Processing page {page_num}/{total_pages}")
    
    if not page_text.strip():
        print(f"  [SKIP] Page {page_num} is empty")
        if job_id:
            update_progress(job_id, message=f"Page {page_num} is empty - skipping")
        return {"page": page_num, "data": None, "status": "empty"}
    
    # Initial structuring with retry logic for malformed JSON
    print(f"  [ITER 1/{iterations}] Initial structuring...")
    if job_id:
        update_progress(job_id, current_iteration=1, message=f"Page {page_num}: Initial structuring...")
    
    current_json = None
    max_retries = 3
    for retry in range(max_retries):
        prompt = _build_ocr_to_json_prompt(page_text)
        if retry > 0:
            # Add stronger instruction on retry
            prompt += "\n\nIMPORTANT: Your previous response had a JSON syntax error. Return ONLY valid JSON with proper commas and brackets."
            print(f"    [RETRY {retry + 1}/{max_retries}] Requesting valid JSON...")
        
        response = _call_qwen(prompt)
        
        try:
            current_json = _extract_json_from_text(response)
            print(f"    [OK] Structured {len(json.dumps(current_json))} chars")
            break  # Success - exit retry loop
        except Exception as e:
            if retry < max_retries - 1:
                print(f"    [WARN] Parse error (retry {retry + 1}): {e}")
            else:
                print(f"    [FAIL] Parse error after {max_retries} attempts: {e}")
                return {"page": page_num, "raw_text": page_text, "error": str(e), "status": "failed"}
    
    # Verification iterations (2 through N)
    for i in range(2, iterations + 1):
        print(f"  [ITER {i}/{iterations}] Verifying completeness...")
        if job_id:
            update_progress(job_id, current_iteration=i, message=f"Page {page_num}: Verification {i}/{iterations}")
        
        prompt = f"""Verify OCR-to-JSON conversion for PAGE {page_num}.

OCR TEXT:
{page_text}

CURRENT JSON:
{json.dumps(current_json, indent=2)}

VERIFICATION CHECKLIST:
1. DATE ACCURACY: Do JSON periods match EXACTLY what's in OCR? Remove invented periods.
2. FREQUENCY: Annual data separate from quarterly/TTM/scenario data?
3. COMPLETENESS: All values for each metric row extracted (not just first/last)?
4. NO MIXING: Best/Worst Case in "scenario_data", TTM in "ttm_data", quarterly in "quarterly_data"?

Fix any issues. Output corrected JSON. Start with {{ and end with }}."""
        
        try:
            response = _call_qwen(prompt)
            new_json = _extract_json_from_text(response)
            
            old_size = len(json.dumps(current_json))
            new_size = len(json.dumps(new_json))
            
            if new_size > old_size:
                print(f"    [OK] Found missing data (+{new_size - old_size} chars)")
                current_json = new_json
            elif new_size < old_size - 100:
                print(f"    [WARN] Data loss detected, keeping previous")
            else:
                print(f"    [OK] Verified complete")
                current_json = new_json
        except Exception as e:
            print(f"    [WARN] Iteration {i} failed: {e}")
    
    # Show the extracted JSON for this page
    print(f"\n  [PAGE {page_num} JSON OUTPUT]")
    print("-" * 50)
    print(json.dumps(current_json, indent=2)[:2000])  # First 2000 chars
    if len(json.dumps(current_json)) > 2000:
        print(f"  ... (truncated, total {len(json.dumps(current_json))} chars)")
    print("-" * 50)
    print(f"  [DONE] Page {page_num} complete: {len(json.dumps(current_json))} chars\n")
    
    # Update progress with page JSON
    if job_id:
        update_progress(job_id, page_json=current_json, 
                       message=f"Page {page_num} complete: {len(json.dumps(current_json))} chars")
    
    return {"page": page_num, "data": current_json, "status": "success"}


def process_pdf_page_by_page(
    pdf_path: str, 
    dpi: int = 200, 
    iterations_per_page: int = 5,
    job_id: str = None
) -> Dict[str, Any]:
    """
    Process PDF page-by-page: OCR each page, LLM structures each, then combine.
    
    Args:
        pdf_path: Path to PDF file
        dpi: DPI for OCR conversion
        iterations_per_page: Verification iterations per page (default 5)
    
    Returns:
        Combined JSON from all pages
    """
    print("\n" + "="*60)
    print("[PAGE-BY-PAGE PROCESSING]")
    print("="*60)
    
    # Initialize progress tracking
    if job_id:
        from app.progress import update_progress
        update_progress(job_id, status="ocr_extraction", message="Starting OCR extraction...")
    
    # Step 1: Extract all pages with OCR
    print("\n[STEP 1] OCR Extraction...")
    page_results = extract_pdf_pages_separately(pdf_path, dpi=dpi)
    total_pages = len(page_results)
    
    if job_id:
        update_progress(job_id, total_pages=total_pages, total_iterations=iterations_per_page,
                       message=f"OCR complete: {total_pages} pages extracted")
    
    # Step 2: Process each page with LLM (5 iterations each)
    print(f"\n[STEP 2] LLM Structuring ({total_pages} pages, {iterations_per_page} iterations each)...")
    
    if job_id:
        update_progress(job_id, status="llm_structuring", message=f"Processing {total_pages} pages...")
    
    all_page_json: List[Dict[str, Any]] = []
    
    for page_data in page_results:
        page_num = page_data["page_num"]
        page_text = page_data["text"]
        
        page_json = structure_single_page(
            page_text, 
            page_num, 
            total_pages, 
            iterations=iterations_per_page,
            job_id=job_id
        )
        all_page_json.append(page_json)
    
    # Step 3: Combine all pages
    print("\n" + "="*60)
    print("[STEP 3] Combining all pages...")
    print("="*60)
    
    combined_data: Dict[str, Any] = {
        "source_file": os.path.basename(pdf_path),
        "total_pages": total_pages,
        "extraction_method": "ocr_page_by_page",
        "pages": {}
    }
    
    successful_pages = 0
    for page_result in all_page_json:
        page_num = page_result["page"]
        status = page_result.get("status", "unknown")
        
        if status == "success" and page_result.get("data"):
            combined_data["pages"][f"page_{page_num}"] = page_result["data"]
            successful_pages += 1
            print(f"  Page {page_num}: {len(json.dumps(page_result['data']))} chars [OK]")
        elif status == "empty":
            combined_data["pages"][f"page_{page_num}"] = {"note": "Empty page"}
            print(f"  Page {page_num}: EMPTY [SKIP]")
        else:
            combined_data["pages"][f"page_{page_num}"] = {
                "error": page_result.get("error", "Unknown error"),
                "raw_text": page_result.get("raw_text", "")[:500]
            }
            print(f"  Page {page_num}: FAILED - {page_result.get('error', 'Unknown')}")
    
    print(f"\n  [OK] Combined {successful_pages}/{total_pages} pages successfully")
    
    if job_id:
        update_progress(job_id, status="merging", message="Merging all pages into final JSON...")
    
    # Step 4: Final verification pass to merge/deduplicate
    print("\n[STEP 4] Final merge verification...")
    
    merge_prompt = f"""You have structured data from {total_pages} pages of a document.
Merge them into a single clean JSON structure, removing duplicates and organizing logically.

PAGE DATA:
{json.dumps(combined_data["pages"], indent=2)}

TASK:
1. Merge all page data into one cohesive structure
2. Remove any duplicate entries
3. Organize by logical categories (e.g., profit_loss, balance_sheet, ratios)
4. Keep ALL unique data - do not lose anything

Output ONLY valid JSON. Start with {{ and end with }}."""
    
    try:
        response = _call_qwen(merge_prompt)
        merged_json = _extract_json_from_text(response)
        combined_data["merged_data"] = merged_json
        print(f"  [OK] Merged into {len(json.dumps(merged_json))} chars")
    except Exception as e:
        print(f"  [WARN] Merge failed: {e}, using page-by-page data")
        combined_data["merged_data"] = combined_data["pages"]
    
    print("\n" + "="*60)
    print(f"[COMPLETE] {total_pages} pages processed")
    print("="*60)
    
    if job_id:
        update_progress(job_id, status="complete", completed=True,
                       message=f"Complete: {total_pages} pages processed successfully")
    
    return combined_data.get("merged_data", combined_data)


def structure_ocr_to_json(ocr_text: str, iterations: int = 3) -> Dict[str, Any]:
    """
    Convert OCR-extracted text to structured JSON (legacy single-call method).
    For multi-page PDFs, use process_pdf_page_by_page instead.
    """
    print(f"\n[STRUCTURING] Converting OCR text to JSON ({iterations} verification passes)...")
    
    # Initial structuring
    print(f"  -> Pass 1/{iterations}: Initial structuring...")
    prompt = _build_ocr_to_json_prompt(ocr_text)
    response = _call_qwen(prompt)
    
    try:
        current_json = _extract_json_from_text(response)
        print(f"    [OK] Structured {len(json.dumps(current_json))} chars of JSON")
    except Exception as e:
        print(f"    [WARN] Failed to parse JSON: {e}")
        current_json = {"raw_text": ocr_text, "error": str(e)}
        return current_json
    
    # Verification passes
    for i in range(2, iterations + 1):
        print(f"  -> Pass {i}/{iterations}: Verifying completeness...")
        
        prompt = f"""Verify OCR-to-JSON conversion quality.

OCR TEXT:
{ocr_text}

CURRENT JSON:
{json.dumps(current_json, indent=2)}

VERIFICATION CHECKLIST:
1. DATE ACCURACY: Do JSON periods match EXACTLY what's in OCR? Remove any invented periods (Mar-13/14/15 if not in source).
2. FREQUENCY: Annual data separate from quarterly/TTM/scenario data? No mixing?
3. COMPLETENESS: All metrics have ALL values? (If 10 columns, 10 values each)
4. CONSISTENCY: All metrics have similar period counts? (sales=10, expenses=10, op=10, etc.)

Fix any issues. Output corrected JSON. Start with {{ and end with }}."""
        
        try:
            response = _call_qwen(prompt)
            new_json = _extract_json_from_text(response)
            
            old_size = len(json.dumps(current_json))
            new_size = len(json.dumps(new_json))
            
            if new_size > old_size:
                print(f"    [OK] Found missing data (+{new_size - old_size} chars)")
                current_json = new_json
            else:
                print(f"    [OK] JSON verified complete")
                current_json = new_json
        except Exception as e:
            print(f"    [WARN] Pass {i} failed: {e}")
    
    return current_json


def extract_with_iterations(doc_text: str, iterations: int = 5) -> Dict[str, Any]:
    """
    Extract data from document with multiple self-check iterations.
    Uses the user's exact prompt approach.
    """
    print(f"\n[EXTRACTION] Starting {iterations}-iteration extraction...")
    
    # Initial extraction using user's exact prompt
    print(f"  -> Iteration 1/{iterations}: Initial extraction...")
    prompt = _build_extraction_prompt(doc_text)
    response = _call_qwen(prompt)
    
    try:
        current_json = _extract_json_from_text(response)
        print(f"    [OK] Extracted {len(json.dumps(current_json))} chars of JSON")
    except Exception as e:
        print(f"    ✗ Failed to parse JSON: {e}")
        current_json = {"raw_response": response, "error": str(e)}
    
    # Self-check iterations (iterations 2 through N)
    for i in range(2, iterations + 1):
        print(f"  -> Iteration {i}/{iterations}: Self-checking for missed data...")
        
        prompt = _build_selfcheck_prompt(
            doc_text, 
            json.dumps(current_json, indent=2),
            i,
            iterations
        )
        
        try:
            response = _call_qwen(prompt)
            new_json = _extract_json_from_text(response)
            
            # Compare sizes to see if new data was found
            old_size = len(json.dumps(current_json))
            new_size = len(json.dumps(new_json))
            
            if new_size > old_size:
                print(f"    [OK] Found additional data (+{new_size - old_size} chars)")
                current_json = new_json
            elif new_size == old_size:
                print(f"    [OK] No additional data found")
                current_json = new_json
            else:
                print(f"    [WARN] JSON size decreased, keeping larger version")
            
        except Exception as e:
            print(f"    [WARN] Iteration {i} failed: {e}, keeping previous result")
    
    return current_json


def analyze_extracted_data(extracted_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze extracted data for the 4 key points:
    1. Type of data (sales report, P&L, etc.)
    2. Number of periods
    3. Type of period (annual, monthly, etc.)
    4. Sample queries to query this data
    """
    print("\n[ANALYSIS] Analyzing extracted data for 4 key points...")
    
    json_str = json.dumps(extracted_json, indent=2)
    
    # Truncate if too long
    if len(json_str) > 20000:
        json_str = json_str[:20000] + "\n... [truncated]"
    
    prompt = _build_analysis_prompt(json_str)
    
    analysis_results = None
    
    for attempt in range(3):  # 3 attempts for analysis
        try:
            print(f"  -> Analysis attempt {attempt + 1}/3...")
            response = _call_qwen(prompt)
            analysis_results = _extract_json_from_text(response)
            
            # Validate required fields (updated to match new prompt)
            required = ["data_type", "number_of_periods", "period_type", "sample_queries"]
            if all(k in analysis_results for k in required):
                print("    [OK] Analysis complete")
                break
            else:
                missing = [k for k in required if k not in analysis_results]
                print(f"    [WARN] Missing fields: {missing}, retrying...")
                
        except Exception as e:
            print(f"    [WARN] Attempt {attempt + 1} failed: {e}")
    
    if analysis_results is None:
        analysis_results = {
            "data_type": {"primary_type": "unknown", "confidence": 0.0},
            "number_of_periods": 0,
            "period_type": "unknown",
            "periods_found": [],
            "sample_queries": [],
            "error": "Analysis failed after 3 attempts"
        }
    
    return analysis_results


def process_document_qwen(
    document_path: str,
    convert_to_pdf_first: bool = False,
    iterations: int = 5,
    use_direct_extraction: bool = True,
    use_paddleocr: bool = False,
    run_audit: bool = True,
    ocr_dpi: int = 200,
    job_id: str = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Main pipeline function for Qwen3 480B processing.
    
    Args:
        document_path: Path to PDF, Excel, or CSV file
        convert_to_pdf_first: If True, converts Excel/CSV to PDF before processing
        iterations: Number of self-check iterations (default 5)
        use_direct_extraction: If True, directly reads Excel/CSV using pandas (recommended)
        use_paddleocr: If True, uses PaddleOCR for PDF extraction (better for scanned docs)
        run_audit: If True, runs audit verification for Excel/CSV extractions
        ocr_dpi: DPI for PaddleOCR conversion (default 200)
    
    Returns:
        Tuple of (extracted_data, analysis)
    """
    print("\n" + "=" * 60)
    print("QWEN3 480B CLOUD - DATA EXTRACTION PIPELINE")
    print("=" * 60)
    
    file_type = _detect_file_type(document_path)
    print(f"\n[INFO] File type: {file_type}")
    print(f"[INFO] File: {document_path}")
    print(f"[INFO] Iterations: {iterations}")
    print(f"[INFO] PaddleOCR: {'enabled' if use_paddleocr else 'disabled'}")
    print(f"[INFO] Audit: {'enabled' if run_audit else 'disabled'}")
    
    audit_result = None
    extraction_method = "pandas"
    
    # For Excel/CSV: Use direct pandas extraction (more accurate)
    if use_direct_extraction and file_type in ["excel", "csv"]:
        print("\n[DIRECT EXTRACTION] Reading structured data with pandas...")
        
        if file_type == "excel":
            extracted_data = _load_excel_as_json(document_path)
            xl = pd.ExcelFile(document_path)
            print(f"  -> Found {len(xl.sheet_names)} sheet(s): {', '.join(xl.sheet_names)}")
            total_rows = sum(info["row_count"] for info in extracted_data["sheets"].values())
            print(f"  -> Total rows: {total_rows}")
        else:  # csv
            extracted_data = _load_csv_as_json(document_path)
            print(f"  -> Loaded {extracted_data['sheets']['data']['row_count']} rows")
        
        # Run audit verification
        if run_audit and AUDIT_AVAILABLE:
            print("\n[AUDIT] Verifying extraction completeness...")
            try:
                audit_result = audit_extraction(document_path, json_data=extracted_data)
                audit_result.print_report()
                
                if not audit_result.passed:
                    print("  [WARNING]  AUDIT FAILED - Some data may be missing!")
                else:
                    print("  [OK] AUDIT PASSED - All data verified")
            except Exception as e:
                print(f"  [WARNING]  Audit failed: {e}")
        
        # Get text representation for LLM analysis
        if file_type == "excel":
            doc_text = _load_excel_text(document_path)
        else:
            doc_text = _load_csv_text(document_path)
        
        # Use LLM only for analysis (data type, periods, queries)
        print("\n[LLM ANALYSIS] Analyzing data structure...")
        analysis = analyze_extracted_data(extracted_data)
        
    else:
        # For PDF or when direct extraction is disabled: Use LLM extraction
        
        # Convert to PDF if requested (for Excel/CSV)
        if convert_to_pdf_first and file_type in ["excel", "csv"]:
            print("\n[CONVERT] Converting to PDF...")
            document_path = convert_to_pdf(document_path)
            file_type = "pdf"
        
        # Load document text
        print("\n[LOAD] Loading document content...")
        
        if file_type == "pdf":
            # Try PaddleOCR if enabled and available
            if use_paddleocr and PADDLEOCR_AVAILABLE:
                print(f"  -> Using OCR with PAGE-BY-PAGE processing")
                print(f"  -> DPI: {ocr_dpi}, Iterations per page: {iterations}")
                extraction_method = "ocr_page_by_page"
                try:
                    # Use page-by-page processing with 5 iterations per page
                    extracted_data = process_pdf_page_by_page(
                        document_path, 
                        dpi=ocr_dpi, 
                        iterations_per_page=iterations,
                        job_id=job_id
                    )
                    analysis = analyze_extracted_data(extracted_data)
                    
                    # Skip the rest of the else block
                    if audit_result:
                        analysis["audit"] = {
                            "passed": audit_result.passed,
                            "source_rows": audit_result.source_row_count,
                            "json_rows": audit_result.json_row_count,
                            "source_numeric_sum": round(audit_result.source_numeric_sum, 2),
                            "json_numeric_sum": round(audit_result.json_numeric_sum, 2),
                            "warnings": audit_result.warnings,
                        }
                    
                    print("\n" + "=" * 60)
                    print("[OK] PROCESSING COMPLETE")
                    print("=" * 60)
                    print(f"\n[SUMMARY]")
                    print(f"  - Extraction Method: {extraction_method}")
                    if "data_type" in analysis:
                        dt = analysis["data_type"]
                        print(f"  - Data Type: {dt.get('primary_type', 'unknown')} (confidence: {dt.get('confidence', 0):.0%})")
                    if "sample_queries" in analysis:
                        print(f"  - Sample Queries: {len(analysis['sample_queries'])} generated")
                    
                    return extracted_data, analysis
                    
                except Exception as e:
                    print(f"  [WARNING] OCR page-by-page failed: {e}")
                    print("  -> Falling back to pdfplumber...")
                    doc_text = _load_pdf_text(document_path)
                    extraction_method = "pdfplumber"
            elif use_paddleocr and not PADDLEOCR_AVAILABLE:
                print("  [WARNING] OCR requested but not installed")
                print("  -> Install: pip install pytesseract pdf2image")
                doc_text = _load_pdf_text(document_path)
                extraction_method = "pdfplumber"
            else:
                doc_text = _load_pdf_text(document_path)
                extraction_method = "pdfplumber"
        elif file_type == "excel":
            doc_text = _load_excel_text(document_path)
            extraction_method = "pandas_text"
        elif file_type == "csv":
            doc_text = _load_csv_text(document_path)
            extraction_method = "pandas_text"
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        print(f"  -> Loaded {len(doc_text)} characters via {extraction_method}")
        
        # LLM extraction for non-OCR cases
        extracted_data = extract_with_iterations(doc_text, iterations)
        
        # Analyze extracted data
        analysis = analyze_extracted_data(extracted_data)
    
    # Add audit result to analysis if available
    if audit_result:
        analysis["audit"] = {
            "passed": audit_result.passed,
            "source_rows": audit_result.source_row_count,
            "json_rows": audit_result.json_row_count,
            "source_numeric_sum": round(audit_result.source_numeric_sum, 2),
            "json_numeric_sum": round(audit_result.json_numeric_sum, 2),
            "warnings": audit_result.warnings,
        }
    
    print("\n" + "=" * 60)
    print("[OK] PROCESSING COMPLETE")
    print("=" * 60)
    
    # Print summary
    print("\n[SUMMARY]")
    print(f"  • Extraction Method: {extraction_method}")
    if "data_type" in analysis:
        dt = analysis["data_type"]
        print(f"  • Data Type: {dt.get('primary_type', 'unknown')} (confidence: {dt.get('confidence', 0):.0%})")
    if "period_analysis" in analysis:
        pa = analysis["period_analysis"]
        print(f"  • Periods: {pa.get('number_of_periods', 0)} {pa.get('period_type', 'unknown')} periods")
    if "sample_queries" in analysis:
        print(f"  • Sample Queries: {len(analysis['sample_queries'])} generated")
    if audit_result:
        status = "[OK] PASSED" if audit_result.passed else "[FAIL] FAILED"
        print(f"  • Audit: {status}")
    
    return extracted_data, analysis


# =========================
# CLI ENTRY POINT
# =========================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.qwen_pipeline <document_path> [options]")
        print("\nOptions:")
        print("  --convert-pdf      Convert Excel/CSV to PDF before processing")
        print("  --iterations N     Number of self-check iterations (default: 5)")
        print("  --paddleocr        Use PaddleOCR for PDF extraction")
        print("  --ocr-dpi N        DPI for OCR conversion (default: 200)")
        print("  --no-audit         Skip audit verification")
        sys.exit(1)
    
    doc_path = sys.argv[1]
    convert_pdf = "--convert-pdf" in sys.argv
    use_paddleocr = "--paddleocr" in sys.argv
    run_audit = "--no-audit" not in sys.argv
    
    iterations = 5
    if "--iterations" in sys.argv:
        idx = sys.argv.index("--iterations")
        if idx + 1 < len(sys.argv):
            iterations = int(sys.argv[idx + 1])
    
    ocr_dpi = 200
    if "--ocr-dpi" in sys.argv:
        idx = sys.argv.index("--ocr-dpi")
        if idx + 1 < len(sys.argv):
            ocr_dpi = int(sys.argv[idx + 1])
    
    extracted, analysis = process_document_qwen(
        doc_path,
        convert_to_pdf_first=convert_pdf,
        iterations=iterations,
        use_paddleocr=use_paddleocr,
        run_audit=run_audit,
        ocr_dpi=ocr_dpi
    )
    
    # Save results
    output_path = os.path.splitext(doc_path)[0] + "_qwen_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "extracted_data": extracted,
            "analysis": analysis
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OUTPUT] Results saved to: {output_path}")
