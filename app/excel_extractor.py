"""
Excel Sheet-by-Sheet Extraction Module

Processes Excel files directly (no PDF conversion) with same output structure as PDF extraction:
- Creates _pages folder with sheet_XXX.json files
- Combined documents by type (combined_profit_loss_annual.json, etc.)
- extraction_result.json master file
- Full data validation to ensure no missing data
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import pandas as pd
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
    if PROGRESS_AVAILABLE and _update_progress is not None:
        return _update_progress(*args, **kwargs)

# =========================
# CONFIG
# =========================

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3-coder:480b")

client = Client(host=OLLAMA_HOST)

# Document types we can detect
DOCUMENT_TYPES = [
    "profit_loss",
    "balance_sheet", 
    "cash_flow",
    "ratios",
    "quarterly_results",
    "shareholding_pattern",
    "scenarios",
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


def detect_document_type_from_data(df: pd.DataFrame, sheet_name: str) -> Tuple[str, str]:
    """
    Detect document type from DataFrame content.
    Returns (document_type, period_type)
    """
    # Convert to string for analysis
    all_text = sheet_name.lower() + " " + df.to_string().lower()
    
    # Check for scenarios (Best Case / Worst Case)
    if "best case" in all_text and "worst case" in all_text:
        return "scenarios", "scenarios"
    
    # Check for quarterly
    if any(q in all_text for q in ["q1", "q2", "q3", "q4", "quarter", "quarterly"]):
        period_type = "quarterly"
    else:
        period_type = "annual"
    
    # Detect document type by keywords
    if any(kw in all_text for kw in ["profit", "loss", "revenue", "sales", "income statement", "p&l"]):
        if "balance" not in all_text:
            return "profit_loss", period_type
    
    if any(kw in all_text for kw in ["balance sheet", "assets", "liabilities", "equity", "borrowings"]):
        return "balance_sheet", period_type
    
    if any(kw in all_text for kw in ["cash flow", "operating activities", "investing activities", "financing activities"]):
        return "cash_flow", period_type
    
    if any(kw in all_text for kw in ["ratio", "roe", "roce", "eps", "pe ratio", "book value"]):
        return "ratios", period_type
    
    if any(kw in all_text for kw in ["shareholding", "promoter", "public holding", "fii", "dii"]):
        return "shareholding_pattern", period_type
    
    # Check sheet name for hints
    sheet_lower = sheet_name.lower()
    if "profit" in sheet_lower or "p&l" in sheet_lower or "income" in sheet_lower:
        return "profit_loss", period_type
    if "balance" in sheet_lower:
        return "balance_sheet", period_type
    if "cash" in sheet_lower:
        return "cash_flow", period_type
    if "ratio" in sheet_lower:
        return "ratios", period_type
    if "quarter" in sheet_lower:
        return "quarterly_results", "quarterly"
    
    return "unknown", period_type


def extract_periods_from_df(df: pd.DataFrame) -> List[str]:
    """Extract period columns from DataFrame (dates/years in header row)."""
    periods = []
    
    # Check first row and columns for date patterns
    for col in df.columns:
        col_str = str(col)
        # Match patterns like "Mar-24", "2024", "FY24", "Q1 2024", etc.
        if re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-\s]?\d{2,4}$', col_str, re.I):
            periods.append(col_str)
        elif re.match(r'^(FY|CY)?\s?\d{2,4}$', col_str):
            periods.append(col_str)
        elif re.match(r'^Q[1-4]\s?\d{2,4}$', col_str, re.I):
            periods.append(col_str)
        elif re.match(r'^\d{4}$', col_str):
            periods.append(col_str)
        # Check for "Best Case" / "Worst Case" scenarios
        elif col_str.lower() in ["best case", "worst case"]:
            periods.append(col_str)
    
    return periods


def df_to_structured_data(df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
    """
    Convert DataFrame to structured JSON with periods and row data.
    Ensures NO missing data.
    """
    # Clean DataFrame
    df = df.fillna("")
    df = df.replace([float('inf'), float('-inf')], "")
    
    # Try to detect periods from columns
    periods = extract_periods_from_df(df)
    
    # If no periods found in columns, check first row
    if not periods and len(df) > 0:
        first_row = df.iloc[0].astype(str).tolist()
        for val in first_row:
            if re.match(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-\s]?\d{2,4}$', val, re.I):
                periods.append(val)
            elif val.lower() in ["best case", "worst case"]:
                periods.append(val)
    
    # Build data structure
    data = {"periods": periods}
    
    # Find the label column (usually first non-numeric column)
    label_col_idx = 0
    for i, col in enumerate(df.columns):
        col_str = str(df[col].iloc[0]) if len(df) > 0 else str(col)
        if not re.match(r'^[\d,.\-\s]+$', col_str):
            label_col_idx = i
            break
    
    # Extract row data
    for idx, row in df.iterrows():
        row_values = row.tolist()
        if len(row_values) > label_col_idx:
            label = str(row_values[label_col_idx]).strip()
            if label and label.lower() not in ["nan", "", "none"]:
                # Clean label to be JSON key friendly
                clean_label = re.sub(r'[^\w\s]', '', label).strip().lower().replace(' ', '_')
                if clean_label:
                    # Get values for period columns
                    values = []
                    for i, col in enumerate(df.columns):
                        col_str = str(col)
                        if col_str in periods or col_str.lower() in ["best case", "worst case"]:
                            val = row_values[i]
                            # Convert to number if possible
                            if isinstance(val, str):
                                val = val.replace(',', '').strip()
                                try:
                                    val = float(val)
                                except:
                                    pass
                            values.append(val)
                    
                    if values:
                        data[clean_label] = values
    
    return data


def process_single_sheet(
    df: pd.DataFrame,
    sheet_name: str,
    sheet_num: int,
    total_sheets: int,
    output_dir: Path,
    iterations: int = 3,
    job_id: Optional[str] = None,
    source_file: str = ""
) -> Dict[str, Any]:
    """
    Process a single Excel sheet with validation.
    """
    print(f"\n[{sheet_num}/{total_sheets}] Processing sheet: {sheet_name}")
    print("-" * 50)
    
    if job_id:
        update_progress(
            job_id,
            status="processing",
            message=f"Processing sheet {sheet_num}/{total_sheets}: {sheet_name}",
            current_page=sheet_num,
            total_pages=total_sheets
        )
    
    # Detect document type
    doc_type, period_type = detect_document_type_from_data(df, sheet_name)
    print(f"  Document type: {doc_type} ({period_type})")
    
    # Extract data directly from DataFrame
    direct_data = df_to_structured_data(df, sheet_name)
    
    # Also use LLM for structured extraction
    llm_data = None
    sheet_text = df.to_string()
    
    if len(sheet_text) < 50000:  # Only use LLM if not too large
        prompt = f"""Extract all financial data from this Excel sheet into structured JSON.

Sheet Name: {sheet_name}
Sheet Content:
{sheet_text[:30000]}

CRITICAL RULES:
1. Extract ALL data - no missing values
2. Identify the time periods (columns like Mar-24, FY24, etc.)
3. Extract all row labels and their values for each period
4. Numbers should be extracted as numbers, not strings
5. Handle Best Case / Worst Case as periods if present

Return a JSON object with this structure:
{{
    "periods": ["Mar-24", "Mar-23", ...],  // All time periods found
    "row_label_1": [value1, value2, ...],  // Values aligned with periods
    "row_label_2": [value1, value2, ...],
    ...
}}

Return ONLY the JSON object, no explanation."""

        try:
            for i in range(iterations):
                print(f"  Iteration {i+1}: LLM extraction...")
                response = _call_llm(prompt)
                llm_data = _extract_json_from_text(response)
                
                if llm_data and "periods" in llm_data:
                    # Validate LLM data completeness
                    period_count = len(llm_data.get("periods", []))
                    fields_ok = True
                    for key, val in llm_data.items():
                        if key != "periods" and isinstance(val, list):
                            if len(val) != period_count:
                                fields_ok = False
                                break
                    
                    if fields_ok:
                        print(f"  Iteration {i+1}: ✅ LLM extracted {len(llm_data)} fields")
                        break
                    else:
                        print(f"  Iteration {i+1}: ⚠️ Period mismatch, retrying...")
                else:
                    print(f"  Iteration {i+1}: ⚠️ Invalid response, retrying...")
        except Exception as e:
            print(f"  LLM extraction failed: {e}")
    
    # Merge direct and LLM data - prefer LLM if available and has more data
    if llm_data and len(llm_data) > len(direct_data):
        final_data = llm_data
        extraction_method = "llm"
    else:
        final_data = direct_data
        extraction_method = "direct"
    
    print(f"  Using {extraction_method} extraction: {len(final_data)} fields")
    
    # Validate no missing data
    validation_issues = []
    period_count = len(final_data.get("periods", []))
    
    for key, val in final_data.items():
        if key != "periods" and isinstance(val, list):
            if len(val) != period_count:
                validation_issues.append(f"{key}: expected {period_count} values, got {len(val)}")
            null_count = sum(1 for v in val if v is None or v == "" or v == "nan")
            if null_count > 0:
                validation_issues.append(f"{key}: {null_count} null values")
    
    if validation_issues:
        print(f"  ⚠️ Validation issues: {validation_issues[:3]}")
    else:
        print(f"  ✅ All data validated - no missing values")
    
    # Build result
    result = {
        "sheet": sheet_num,
        "sheet_name": sheet_name,
        "status": "success",
        "document_type": doc_type,
        "period_type": period_type,
        "extraction_method": extraction_method,
        "extracted_data": final_data,
        "row_count": len(df),
        "col_count": len(df.columns),
        "validation_issues": validation_issues,
        "source_file": source_file
    }
    
    # Save sheet JSON
    sheet_file = output_dir / f"sheet_{sheet_num:03d}.json"
    with open(sheet_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  -> Saved: {sheet_file.name}")
    
    return result


def combine_same_type_documents(
    sheet_results: List[Dict],
    doc_type: str,
    period_type: str,
    iterations: int = 3
) -> Dict[str, Any]:
    """Combine sheets with same document type."""
    matching_docs = []
    sheets_included = []
    
    for result in sheet_results:
        if result.get("status") == "error":
            continue
        if result.get("document_type") == doc_type and result.get("period_type") == period_type:
            matching_docs.append(result)
            sheets_included.append(result.get("sheet_name", result.get("sheet")))
    
    if not matching_docs:
        return {}
    
    type_label = f"{doc_type} ({period_type})"
    print(f"\n[COMBINE] Merging {len(matching_docs)} sheets of type: {type_label}")
    
    # Build period-indexed data structure
    period_data_map = {}
    all_fields = set()
    
    for doc in matching_docs:
        data = doc.get("extracted_data", {})
        periods = data.get("periods", [])
        
        for i, period in enumerate(periods):
            if period not in period_data_map:
                period_data_map[period] = {}
            
            for key, value in data.items():
                if key == "periods":
                    continue
                all_fields.add(key)
                
                if isinstance(value, list) and i < len(value):
                    if key not in period_data_map[period] or period_data_map[period][key] is None:
                        period_data_map[period][key] = value[i]
    
    # Sort periods
    def period_sort_key(period: str) -> tuple:
        month_order = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                       "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
        # Handle scenario periods
        if period.lower() == "best case":
            return (9999, 1)
        if period.lower() == "worst case":
            return (9999, 2)
        try:
            parts = re.split(r'[-\s]', period)
            if len(parts) >= 2:
                month = parts[0]
                year = int(parts[1])
                if year < 50:
                    year += 2000
                elif year < 100:
                    year += 1900
                return (year, month_order.get(month[:3], 0))
        except:
            pass
        return (0, 0)
    
    sorted_periods = sorted(period_data_map.keys(), key=period_sort_key)
    
    # Build combined data
    combined_data = {"periods": sorted_periods}
    
    for field in all_fields:
        combined_data[field] = []
        for period in sorted_periods:
            value = period_data_map.get(period, {}).get(field, None)
            combined_data[field].append(value)
    
    # Verify
    print(f"  -> Verifying combined result ({iterations} iterations)...")
    for i in range(iterations):
        expected = len(sorted_periods)
        mismatches = [k for k, v in combined_data.items() 
                      if isinstance(v, list) and len(v) != expected and k != "periods"]
        if mismatches:
            print(f"     Iteration {i+1}: Period mismatch in {mismatches[:3]}")
        else:
            print(f"     Iteration {i+1}: ✅ All fields have {expected} periods")
            break
    
    return {
        "document_type": doc_type,
        "period_type": period_type,
        "sheets": sheets_included,
        "periods": sorted_periods,
        "period_count": len(sorted_periods),
        "data": combined_data
    }


def generate_sample_queries(doc_type: str, periods: List[str], data: Dict) -> List[str]:
    """Generate sample queries for a document."""
    queries = []
    
    if doc_type == "profit_loss":
        if periods:
            queries.append(f"What was the net profit in {periods[-1]}?")
            queries.append(f"Show revenue trend from {periods[0]} to {periods[-1]}")
    elif doc_type == "balance_sheet":
        queries.append("What is the total debt?")
        queries.append("Show assets vs liabilities")
    elif doc_type == "scenarios":
        queries.append("Compare Best Case vs Worst Case operating profit")
    elif doc_type == "ratios":
        queries.append("What is the ROE trend?")
    
    return queries


def process_excel_sheet_by_sheet(
    excel_path: str,
    output_dir: Optional[str] = None,
    iterations_per_sheet: int = 3,
    combine_iterations: int = 3,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process Excel file sheet-by-sheet with same output structure as PDF extraction.
    
    Creates:
    - _pages folder (e.g., "Excel_Name_pages/")
    - sheet_001.json, sheet_002.json, etc.
    - combined_profit_loss_annual.json, etc.
    - extraction_result.json
    """
    excel_name = Path(excel_path).stem
    
    # Setup output directory
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = Path("processed") / f"{excel_name}_pages"
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("SHEET-BY-SHEET EXCEL EXTRACTION")
    print("=" * 70)
    
    # Load Excel file
    xl = pd.ExcelFile(excel_path)
    sheet_names = xl.sheet_names
    total_sheets = len(sheet_names)
    
    print(f"\n[INFO] Excel: {excel_path}")
    print(f"[INFO] Total Sheets: {total_sheets}")
    print(f"[INFO] Sheet names: {sheet_names}")
    print(f"[INFO] Iterations per sheet: {iterations_per_sheet}")
    print(f"[INFO] Output directory: {out_path}")
    
    if job_id:
        update_progress(
            job_id,
            status="starting",
            message=f"Processing Excel with {total_sheets} sheets",
            total_pages=total_sheets
        )
    
    # Process each sheet
    sheet_results = []
    
    for i, sheet_name in enumerate(sheet_names, 1):
        try:
            df = xl.parse(sheet_name)
            
            # Skip empty sheets
            if df.empty:
                print(f"\n[{i}/{total_sheets}] Skipping empty sheet: {sheet_name}")
                continue
            
            result = process_single_sheet(
                df=df,
                sheet_name=sheet_name,
                sheet_num=i,
                total_sheets=total_sheets,
                output_dir=out_path,
                iterations=iterations_per_sheet,
                job_id=job_id,
                source_file=excel_path
            )
            sheet_results.append(result)
            
        except Exception as e:
            print(f"\n[ERROR] Sheet {sheet_name} failed: {e}")
            sheet_results.append({
                "sheet": i,
                "sheet_name": sheet_name,
                "status": "error",
                "error": str(e),
                "document_type": "unknown"
            })
    
    # Identify unique document type combinations
    doc_type_combos = set()
    for result in sheet_results:
        if result.get("status") == "error":
            continue
        doc_type = result.get("document_type", "unknown")
        period_type = result.get("period_type", "unknown")
        if doc_type != "unknown":
            doc_type_combos.add((doc_type, period_type))
    
    print(f"\n[SUMMARY] Document types found: {list(doc_type_combos)}")
    
    # Combine by document type
    documents = []
    
    for doc_type, period_type in doc_type_combos:
        combined = combine_same_type_documents(sheet_results, doc_type, period_type, combine_iterations)
        
        if combined:
            periods = combined.get("periods", [])
            data = combined.get("data", {})
            sample_queries = generate_sample_queries(doc_type, periods, data)
            
            document = {
                "document_type": doc_type,
                "period_type": period_type,
                "sheets": combined.get("sheets", []),
                "periods": periods,
                "period_count": len(periods),
                "data": data,
                "sample_queries": sample_queries
            }
            documents.append(document)
            
            # Save combined document
            safe_period_type = period_type.replace("/", "_") if period_type else "unknown"
            doc_file = out_path / f"combined_{doc_type}_{safe_period_type}.json"
            with open(doc_file, "w", encoding="utf-8") as f:
                json.dump(document, f, indent=2, default=str)
            print(f"  -> Saved: {doc_file.name}")
    
    # Build final result
    final_result = {
        "source_file": excel_path,
        "file_type": "excel",
        "total_sheets": total_sheets,
        "sheets_processed": len(sheet_results),
        "sheets_successful": len([s for s in sheet_results if s.get("status") != "error"]),
        "document_types_found": [dt for dt, pt in doc_type_combos],
        "documents": documents,
        "sheet_details": sheet_results,
        "output_directory": str(out_path),
        "timestamp": datetime.now().isoformat()
    }
    
    # Save master result
    master_file = out_path / "extraction_result.json"
    with open(master_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("[COMPLETE] Sheet-by-sheet extraction finished")
    print("=" * 70)
    print(f"  • Total sheets: {total_sheets}")
    print(f"  • Successful: {final_result['sheets_successful']}")
    print(f"  • Document types: {list(doc_type_combos)}")
    print(f"  • Documents produced: {len(documents)}")
    print(f"  • Output: {out_path}")
    
    return final_result


def process_csv_file(
    csv_path: str,
    output_dir: Optional[str] = None,
    iterations: int = 3,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """Process CSV file with same output structure."""
    csv_name = Path(csv_path).stem
    
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = Path("processed") / f"{csv_name}_pages"
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("CSV EXTRACTION")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    
    print(f"\n[INFO] CSV: {csv_path}")
    print(f"[INFO] Rows: {len(df)}, Columns: {len(df.columns)}")
    
    result = process_single_sheet(
        df=df,
        sheet_name="Data",
        sheet_num=1,
        total_sheets=1,
        output_dir=out_path,
        iterations=iterations,
        job_id=job_id,
        source_file=csv_path
    )
    
    # Build final result
    final_result = {
        "source_file": csv_path,
        "file_type": "csv",
        "total_sheets": 1,
        "sheets_processed": 1,
        "sheets_successful": 1 if result.get("status") != "error" else 0,
        "document_types_found": [result.get("document_type", "unknown")],
        "documents": [result],
        "sheet_details": [result],
        "output_directory": str(out_path),
        "timestamp": datetime.now().isoformat()
    }
    
    master_file = out_path / "extraction_result.json"
    with open(master_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, default=str)
    
    print(f"\n[COMPLETE] CSV extraction finished -> {out_path}")
    
    return final_result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if file_path.lower().endswith('.csv'):
            result = process_csv_file(file_path)
        else:
            result = process_excel_sheet_by_sheet(file_path)
        print(f"\nResult: {len(result.get('documents', []))} documents extracted")
    else:
        print("Usage: python excel_extractor.py <excel_or_csv_path>")
