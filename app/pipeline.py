import json
import os
import re
from typing import Dict, Any, Tuple

import pandas as pd
import pdfplumber
import requests
from openai import OpenAI

from .extraction_templates import TEMPLATE_MAP, GENERIC_TEMPLATE, PROFIT_LOSS_TEMPLATE


# =========================
# CONFIG
# =========================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

USE_OPENAI_FALLBACK = os.getenv("USE_OPENAI_FALLBACK", "false").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

print(f"Ollama Host: {OLLAMA_HOST}")
print(f"Ollama Model: {OLLAMA_MODEL}")


# =========================
# JSON UTIL
# =========================

def _extract_json_from_text(text: str) -> Any:
    fenced = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first:last + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        return json.loads(text)


# =========================
# PERIOD DETECTION
# =========================

MONTH_ALIASES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

YEAR_RE = re.compile(r"\b(20\d{2})\b")
QUARTER_RE = re.compile(r"\b(Q[1-4])\s*(FY)?\s*(\d{2,4})\b", re.IGNORECASE)
MONTH_YEAR_RE = re.compile(
    r"\b("
    r"jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|"
    r"jun(e)?|jul(y)?|aug(ust)?"
    r"|sep(t)?(ember)?|oct(ober)?|nov(ember)?|dec(ember)?"
    r")\s*[-'/]?\s*(20\d{2}|\d{2})\b",
    re.IGNORECASE
)
NUMERIC_MONTH_RE = re.compile(r"\b(0?[1-9]|1[0-2])\s*[-/.]\s*(20\d{2})\b")


def detect_periods(text: str) -> Dict[str, Any]:
    """Detect daily, weekly, monthly, quarterly, and yearly periods in text."""
    raw_years = set(YEAR_RE.findall(text))
    years = []
    for y in raw_years:
        try:
            val = int(y)
            if 1980 <= val <= 2035:
                years.append(y)
        except ValueError:
            pass
    years = sorted(years)
    
    quarters = set()
    months = set()
    weeks = set()
    days = set()

    # Quarterly
    for q, _, y in QUARTER_RE.findall(text):
        quarters.add(f"{q.upper()} FY{y}")

    # Monthly
    for m, _, y in MONTH_YEAR_RE.findall(text):
        month_num = MONTH_ALIASES[m.lower()[:3]]
        year = f"20{y}" if len(y) == 2 else y
        months.add(f"{year}-{month_num:02d}")

    for m, y in NUMERIC_MONTH_RE.findall(text):
        months.add(f"{y}-{int(m):02d}")
    
    # Weekly
    week_pattern = re.compile(r"\b(W|Week)\s*(\d{1,2})\b", re.IGNORECASE)
    for match in week_pattern.findall(text):
        weeks.add(f"Week {match[1]}")
    
    # Daily
    daily_pattern = re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
    for match in daily_pattern.findall(text):
        days.add(f"{match[0]}-{match[1]}-{match[2]}")

    # Determine type
    if days:
        period_type = "daily" if not (weeks or months or quarters) else "mixed"
    elif weeks:
        period_type = "weekly" if not (months or quarters) else "mixed"
    elif months and (quarters or years):
        period_type = "mixed"
    elif months:
        period_type = "monthly"
    elif quarters:
        period_type = "quarterly"
    elif years:
        period_type = "annual"
    else:
        period_type = "unknown"

    return {
        "period_type": period_type,
        "years": years,
        "quarters": sorted(quarters),
        "months": sorted(months),
        "weeks": sorted(weeks),
        "days": sorted(days)[:50],
        "confidence_score": 0.9 if period_type != "unknown" else 0.3
    }


# =========================
# FILE LOADERS
# =========================

def _detect_file_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext in [".xls", ".xlsx"]:
        return "excel"
    return "unknown"


def _load_pdf_text(path: str) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            parts.append(f"--- PAGE {i+1} ---\n{page.extract_text() or ''}")
    return "\n\n".join(parts)


def _load_excel_as_markdown(path: str) -> str:
    """Load Excel file and convert each sheet to Markdown table."""
    try:
        import tabulate
    except ImportError:
        print("  → Installing tabulate...")
        os.system("pip install tabulate")
        import tabulate
    
    xl = pd.ExcelFile(path)
    parts = []
    
    print(f"  → Found {len(xl.sheet_names)} sheets: {', '.join(xl.sheet_names)}")
    
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        
        # Skip empty sheets
        if df.empty or df.isna().all().all():
            print(f"  → Skipping empty sheet: {sheet_name}")
            continue
        
        print(f"  → Processing sheet: {sheet_name} ({len(df)} rows x {len(df.columns)} cols)")
        
        # Convert to markdown
        markdown = df.fillna("").to_markdown(index=False)
        
        parts.append(f"### SHEET: {sheet_name} ###\n{markdown}\n")
    
    print(f"  → Total content size: {len(''.join(parts))} characters")
    return "\n\n".join(parts)


# =========================
# LLM CALLER
# =========================

def _call_llm(system_prompt: str, user_prompt: str) -> Tuple[str, str]:
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/v1/chat/completions",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"}
            },
            timeout=600
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"], OLLAMA_MODEL
    except Exception as e:
        if USE_OPENAI_FALLBACK:
            print(f"Ollama failed ({e}), using OpenAI fallback...")
            client = OpenAI()
            completion = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            return completion.choices[0].message.content, OPENAI_MODEL
        raise


# =========================
# PROMPTS
# =========================

def _build_classification_prompt(doc_text: str) -> str:
    return f"""
Classify this document.

DOCUMENT CONTENT:
{doc_text[:3000]}

Output JSON with:
- document_type: "profit_and_loss", "balance_sheet", "cash_flow", or "generic"
- confidence_score: 0.0 to 1.0
- reasoning: brief explanation

JSON OUTPUT:
"""


def _build_extraction_prompt(doc_text: str) -> str:
    return f"""
Extract ALL financial data from this document.

CRITICAL INSTRUCTIONS:
1. This document may contain MULTIPLE TABLES in a single sheet
2. Identify each table by looking for section headers like:
   - "PROFIT & LOSS" or "PROFIT AND LOSS"
   - "BALANCE SHEET"
   - "CASH FLOW"
   - "QUARTERS" or "QUARTERLY"
3. For EACH table found:
   - Look for "Report Date" row to identify date columns
   - Extract ALL rows of data
   - Separate annual data from quarterly data
4. Output in this structure:

TEMPLATE:
{json.dumps(PROFIT_LOSS_TEMPLATE, indent=2)}

DOCUMENT CONTENT:
{doc_text}

JSON OUTPUT:
"""


# =========================
# MAIN PIPELINE
# =========================

def process_document(document_path: str, template_path: str = None) -> Tuple[dict, str]:
    """
    Simple 4-step process:
    1. Classification
    2. Period Detection
    3. Data Extraction (LLM detects tables)
    4. Return structured JSON
    """
    print("\n" + "="*60)
    print("STARTING DOCUMENT PROCESSING")
    print("="*60)
    
    file_type = _detect_file_type(document_path)
    print(f"File type: {file_type}")
    
    # Load document
    print(f"\n[STEP 0] Loading {file_type} file...")
    if file_type == "pdf":
        doc_text = _load_pdf_text(document_path)
        print(f"  → Loaded PDF: {len(doc_text)} characters")
    elif file_type == "excel":
        doc_text = _load_excel_as_markdown(document_path)
    else:
        raise ValueError("Unsupported file type")

    # Step 1: Classification
    print(f"\n[STEP 1] Classifying document...")
    try:
        class_resp, _ = _call_llm(
            "Output strict JSON only.",
            _build_classification_prompt(doc_text)
        )
        class_json = _extract_json_from_text(class_resp)
        doc_type = class_json.get("document_type", "generic")
        confidence = class_json.get("confidence_score", 0)
        reasoning = class_json.get("reasoning", "N/A")
        print(f"  ✓ Type: {doc_type}")
        print(f"  ✓ Confidence: {confidence}")
        print(f"  ✓ Reasoning: {reasoning}")
    except Exception as e:
        print(f"  ✗ Classification failed: {e}")
        class_json = {
            "document_type": "generic",
            "confidence_score": 0.0,
            "reasoning": "Classification failed"
        }
        doc_type = "generic"

    # Step 2: Period Detection
    print(f"\n[STEP 2] Detecting periods...")
    period_info = detect_periods(doc_text)
    print(f"  ✓ Period Type: {period_info['period_type']}")
    if period_info['years']:
        print(f"  ✓ Years: {len(period_info['years'])} found - {', '.join(period_info['years'][:5])}{'...' if len(period_info['years']) > 5 else ''}")
    if period_info['quarters']:
        print(f"  ✓ Quarters: {len(period_info['quarters'])} found - {', '.join(period_info['quarters'][:3])}{'...' if len(period_info['quarters']) > 3 else ''}")
    if period_info['months']:
        print(f"  ✓ Months: {len(period_info['months'])} found - {', '.join(period_info['months'][:3])}{'...' if len(period_info['months']) > 3 else ''}")
    if period_info['weeks']:
        print(f"  ✓ Weeks: {len(period_info['weeks'])} found - {', '.join(period_info['weeks'][:3])}{'...' if len(period_info['weeks']) > 3 else ''}")
    if period_info['days']:
        print(f"  ✓ Days: {len(period_info['days'])} found - {', '.join(period_info['days'][:3])}{'...' if len(period_info['days']) > 3 else ''}")

    # Step 3: Data Extraction
    print(f"\n[STEP 3] Extracting data...")
    print(f"  → Sending to LLM (this may take 30-60 seconds)...")
    try:
        raw, model_used = _call_llm(
            "Output strict JSON only. Detect and extract ALL tables in the document.",
            _build_extraction_prompt(doc_text)
        )
        print(f"  ✓ LLM response received ({len(raw)} characters)")
        print(f"  → Parsing JSON...")
        json_result = _extract_json_from_text(raw)
        print(f"  ✓ Extraction complete using {model_used}")
        
        # Show what was extracted
        if isinstance(json_result, dict) and "companies" in json_result:
            print(f"  ✓ Found {len(json_result['companies'])} company/companies")
            for i, company in enumerate(json_result['companies'], 1):
                print(f"    Company {i}: {company.get('company_name', 'UNKNOWN')}")
                if 'profit_loss' in company:
                    pl = company['profit_loss']
                    if 'annual' in pl and pl['annual']:
                        print(f"      → P&L Annual: {len(pl['annual'])} periods")
                    if 'quarters' in pl and 'quarterly' in pl['quarters'] and pl['quarters']['quarterly']:
                        print(f"      → P&L Quarterly: {len(pl['quarters']['quarterly'])} periods")
                    if 'monthly' in pl and pl['monthly']:
                        print(f"      → P&L Monthly: {len(pl['monthly'])} periods")
                    if 'weekly' in pl and pl['weekly']:
                        print(f"      → P&L Weekly: {len(pl['weekly'])} periods")
                    if 'daily' in pl and pl['daily']:
                        print(f"      → P&L Daily: {len(pl['daily'])} periods")
    except Exception as e:
        print(f"  ✗ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        json_result = {"error": str(e)}
        model_used = "none"

    # Step 4: Add metadata
    print(f"\n[STEP 4] Adding metadata...")
    if isinstance(json_result, dict):
        json_result.setdefault("metadata", {})
        json_result["metadata"]["classification"] = class_json
        json_result["metadata"]["periods"] = period_info
        print(f"  ✓ Metadata injected")
    else:
        json_result = {
            "data": json_result,
            "metadata": {
                "classification": class_json,
                "periods": period_info
            }
        }
        print(f"  ✓ Result wrapped with metadata")

    print("\n" + "="*60)
    print("✓ PROCESSING COMPLETE")
    print("="*60 + "\n")
    
    return json_result, model_used
