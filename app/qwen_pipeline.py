"""
Qwen3 480B Cloud Pipeline for Data Extraction

This module uses Ollama's Qwen3 480B cloud model for:
1. PDF/Excel/CSV data extraction with 5-iteration self-check
2. Document type classification
3. Period detection and analysis
4. Sample query generation
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
    print(f"  → Converted to PDF: {output_path}")
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
USER_EXTRACTION_PROMPT = """You are a data analyst. Extract all data from pdf in json format make sure you get all the data. I want you to iterate at least 5 times to self check if you have missed any thing at all from the pdf. And if missed, make sure that missed data gets into the json."""


def _build_extraction_prompt(doc_text: str) -> str:
    """Build the extraction prompt using the user's exact prompt."""
    return f"""{USER_EXTRACTION_PROMPT}

CRITICAL REQUIREMENTS:
1. For tabular data, include the column headers/periods with each value. Use this format:
   "Sales": [{{"period": "Mar-16", "value": 6898.23}}, {{"period": "Mar-17", "value": 7689.37}}, ...]
2. Keep percentage values as strings like "27%" or "12.13%"
3. Keep "-" or empty cells as null
4. Extract ALL rows, ALL columns, ALL values exactly as shown
5. Include company name, source, ratios, trends - everything

DOCUMENT CONTENT:
{doc_text}

Output ONLY valid JSON. Start with {{ and end with }}."""


def _build_selfcheck_prompt(doc_text: str, current_json: str, iteration: int, total_iterations: int) -> str:
    """Build prompt for self-checking extracted data - using user's iteration approach."""
    return f"""You are a data analyst. This is self-check iteration {iteration} of {total_iterations}.

You previously extracted data from a document. Now verify your extraction is COMPLETE.

ORIGINAL DOCUMENT:
{doc_text}

YOUR CURRENT EXTRACTION:
{current_json}

TASK - SELF CHECK ITERATION {iteration}/{total_iterations}:
1. Compare your extraction with the original document
2. Check if ANY data was missed - every single number, every row, every column
3. If you find ANYTHING missing, add it to the JSON
4. If you find errors, correct them
5. Make sure you get ALL the data

Output the COMPLETE JSON with any corrections. Start with {{ and end with }}."""


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

def extract_with_iterations(doc_text: str, iterations: int = 5) -> Dict[str, Any]:
    """
    Extract data from document with multiple self-check iterations.
    Uses the user's exact prompt approach.
    """
    print(f"\n[EXTRACTION] Starting {iterations}-iteration extraction...")
    
    # Initial extraction using user's exact prompt
    print(f"  → Iteration 1/{iterations}: Initial extraction...")
    prompt = _build_extraction_prompt(doc_text)
    response = _call_qwen(prompt)
    
    try:
        current_json = _extract_json_from_text(response)
        print(f"    ✓ Extracted {len(json.dumps(current_json))} chars of JSON")
    except Exception as e:
        print(f"    ✗ Failed to parse JSON: {e}")
        current_json = {"raw_response": response, "error": str(e)}
    
    # Self-check iterations (iterations 2 through N)
    for i in range(2, iterations + 1):
        print(f"  → Iteration {i}/{iterations}: Self-checking for missed data...")
        
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
                print(f"    ✓ Found additional data (+{new_size - old_size} chars)")
                current_json = new_json
            elif new_size == old_size:
                print(f"    ✓ No additional data found")
                current_json = new_json
            else:
                print(f"    ⚠ JSON size decreased, keeping larger version")
            
        except Exception as e:
            print(f"    ⚠ Iteration {i} failed: {e}, keeping previous result")
    
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
            print(f"  → Analysis attempt {attempt + 1}/3...")
            response = _call_qwen(prompt)
            analysis_results = _extract_json_from_text(response)
            
            # Validate required fields (updated to match new prompt)
            required = ["data_type", "number_of_periods", "period_type", "sample_queries"]
            if all(k in analysis_results for k in required):
                print("    ✓ Analysis complete")
                break
            else:
                missing = [k for k in required if k not in analysis_results]
                print(f"    ⚠ Missing fields: {missing}, retrying...")
                
        except Exception as e:
            print(f"    ⚠ Attempt {attempt + 1} failed: {e}")
    
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
    use_direct_extraction: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Main pipeline function for Qwen3 480B processing.
    
    Args:
        document_path: Path to PDF, Excel, or CSV file
        convert_to_pdf_first: If True, converts Excel/CSV to PDF before processing
        iterations: Number of self-check iterations (default 5)
        use_direct_extraction: If True, directly reads Excel/CSV using pandas (recommended)
    
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
    
    # For Excel/CSV: Use direct pandas extraction (more accurate)
    if use_direct_extraction and file_type in ["excel", "csv"]:
        print("\n[DIRECT EXTRACTION] Reading structured data with pandas...")
        
        if file_type == "excel":
            extracted_data = _load_excel_as_json(document_path)
            xl = pd.ExcelFile(document_path)
            print(f"  → Found {len(xl.sheet_names)} sheet(s): {', '.join(xl.sheet_names)}")
            total_rows = sum(info["row_count"] for info in extracted_data["sheets"].values())
            print(f"  → Total rows: {total_rows}")
        else:  # csv
            extracted_data = _load_csv_as_json(document_path)
            print(f"  → Loaded {extracted_data['sheets']['data']['row_count']} rows")
        
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
            doc_text = _load_pdf_text(document_path)
        elif file_type == "excel":
            doc_text = _load_excel_text(document_path)
        elif file_type == "csv":
            doc_text = _load_csv_text(document_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        print(f"  → Loaded {len(doc_text)} characters")
        
        # Extract data with iterations
        extracted_data = extract_with_iterations(doc_text, iterations)
        
        # Analyze extracted data
        analysis = analyze_extracted_data(extracted_data)
    
    # Combine results
    final_result = {
        "extracted_data": extracted_data,
        "analysis": analysis,
        "metadata": {
            "source_file": os.path.basename(document_path),
            "file_type": file_type,
            "extraction_iterations": iterations,
            "model": QWEN_MODEL,
            "processed_at": datetime.now().isoformat()
        }
    }
    
    print("\n" + "=" * 60)
    print("✓ PROCESSING COMPLETE")
    print("=" * 60)
    
    # Print summary
    print("\n[SUMMARY]")
    if "data_type" in analysis:
        dt = analysis["data_type"]
        print(f"  • Data Type: {dt.get('primary_type', 'unknown')} (confidence: {dt.get('confidence', 0):.0%})")
    if "period_analysis" in analysis:
        pa = analysis["period_analysis"]
        print(f"  • Periods: {pa.get('number_of_periods', 0)} {pa.get('period_type', 'unknown')} periods")
    if "sample_queries" in analysis:
        print(f"  • Sample Queries: {len(analysis['sample_queries'])} generated")
    
    return extracted_data, analysis


# =========================
# CLI ENTRY POINT
# =========================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.qwen_pipeline <document_path> [--convert-pdf] [--iterations N]")
        sys.exit(1)
    
    doc_path = sys.argv[1]
    convert_pdf = "--convert-pdf" in sys.argv
    
    iterations = 5
    if "--iterations" in sys.argv:
        idx = sys.argv.index("--iterations")
        if idx + 1 < len(sys.argv):
            iterations = int(sys.argv[idx + 1])
    
    extracted, analysis = process_document_qwen(
        doc_path,
        convert_to_pdf_first=convert_pdf,
        iterations=iterations
    )
    
    # Save results
    output_path = os.path.splitext(doc_path)[0] + "_qwen_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "extracted_data": extracted,
            "analysis": analysis
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OUTPUT] Results saved to: {output_path}")
