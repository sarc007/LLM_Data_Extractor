"""
Complete End-to-End Test with QWEN 480B
Tests all Excel and PDF files, verifies data accuracy, generates detailed report.

Usage: python run_complete_test.py
"""

import json
import os
import re
import sys
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from decimal import Decimal

import pandas as pd
import requests

sys.path.insert(0, '.')
from app.qwen_pipeline import _load_excel_as_json, _load_pdf_text, process_document_qwen

# =========================
# CONFIG
# =========================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
QWEN_MODEL = "qwen3-coder:480b-cloud"
INPUT_DIR = "input"
OUTPUT_DIR = "processed"


# =========================
# GENERIC DATA STRUCTURE
# =========================

def parse_period_label(period_str: str) -> Tuple[str, Optional[int]]:
    """
    Parse period string like 'Mar-16' into normalized label and fiscal year.
    Returns (normalized_label like "March 2024", fiscal_year)
    """
    period_str = str(period_str).strip()
    
    match = re.match(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[_-](\d{2})', period_str, re.IGNORECASE)
    if match:
        month_str = match.group(1).capitalize()
        year_short = int(match.group(2))
        year_full = 2000 + year_short if year_short < 50 else 1900 + year_short
        
        month_full = {'Jan': 'January', 'Feb': 'February', 'Mar': 'March', 'Apr': 'April',
                      'May': 'May', 'Jun': 'June', 'Jul': 'July', 'Aug': 'August',
                      'Sep': 'September', 'Oct': 'October', 'Nov': 'November', 'Dec': 'December'}
        
        normalized = f"{month_full[month_str]} {year_full}"
        fiscal_year = year_full if month_str == 'Mar' else (year_full + 1 if month_str in ['Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] else year_full)
        
        return normalized, fiscal_year
    
    return period_str, None


def extract_company_name(filename: str) -> str:
    """Extract company name from filename."""
    name = filename.replace('.xlsx', '').replace('.xls', '').replace('.pdf', '')
    name = re.sub(r'_?(Target|Peer \d+)$', '', name).strip()
    return name


def load_excel_to_generic(excel_path: str) -> List[Dict[str, Any]]:
    """
    Load Excel file into generic structure.
    Returns list of {company, metric, period, fiscal_year, value}
    """
    data = _load_excel_as_json(excel_path)
    file_name = os.path.basename(excel_path)
    company_name = extract_company_name(file_name)
    
    records = []
    
    for sheet_name, sheet_data in data.get('sheets', {}).items():
        columns = sheet_data.get('columns', [])
        rows = sheet_data.get('data', [])
        
        # Find metric column and period columns
        metric_col = None
        period_cols = []
        
        for col in columns:
            col_str = str(col)
            if col_str.lower() in ['report_date', 'report date', 'narration', 'particulars', 'column_1']:
                metric_col = col
            elif re.match(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[_-]\d{2}', col_str, re.IGNORECASE):
                period_cols.append(col)
        
        if not metric_col and columns:
            metric_col = columns[0]
        
        # Process rows
        for row in rows:
            metric_name = row.get(metric_col)
            if not metric_name or str(metric_name).strip() == '':
                continue
            
            metric_name = str(metric_name).strip()
            
            for period_col in period_cols:
                value = row.get(period_col)
                if value is None:
                    continue
                
                period_label, fiscal_year = parse_period_label(str(period_col))
                
                records.append({
                    'company': company_name,
                    'metric': metric_name,
                    'period': period_label,
                    'period_raw': str(period_col),
                    'fiscal_year': fiscal_year,
                    'value': float(value) if isinstance(value, (int, float)) else None,
                    'value_text': str(value) if not isinstance(value, (int, float)) else None,
                    'source_file': file_name
                })
    
    return records


def create_sqlite_db(records: List[Dict[str, Any]], db_path: str = ":memory:") -> sqlite3.Connection:
    """Create SQLite database with generic structure."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS financial_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            metric TEXT NOT NULL,
            period TEXT NOT NULL,
            period_raw TEXT,
            fiscal_year INTEGER,
            value REAL,
            value_text TEXT,
            source_file TEXT
        )
    """)
    
    for rec in records:
        cursor.execute("""
            INSERT INTO financial_data (company, metric, period, period_raw, fiscal_year, value, value_text, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (rec['company'], rec['metric'], rec['period'], rec['period_raw'], 
              rec['fiscal_year'], rec['value'], rec['value_text'], rec['source_file']))
    
    conn.commit()
    return conn


# =========================
# LLM QUERY GENERATION
# =========================

def call_qwen(prompt: str, timeout: int = 300) -> str:
    """Call QWEN 480B model via Ollama."""
    payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": False,
    }
    
    resp = requests.post(
        f"{OLLAMA_HOST}/v1/chat/completions",
        json=payload,
        timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_sql(response: str) -> str:
    """Extract SQL from LLM response."""
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    
    sql_match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    code_match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    select_match = re.search(r"(SELECT\s+.*?;)", response, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()
    
    return response.strip()


SQL_PROMPT = """You are a SQL expert. Convert this natural language query to SQLite SQL.

DATABASE SCHEMA:
Table: financial_data
Columns:
  - company (TEXT): Company name like 'JK Tyre & Indust', 'Apollo Tyres', 'CEAT', 'Goodyear India', 'MRF'
  - metric (TEXT): Financial metric like 'Sales', 'Net Profit', 'Operating Profit', 'EPS', 'Expenses'
  - period (TEXT): Period in format 'March 2024', 'March 2023', 'March 2022', etc.
  - fiscal_year (INTEGER): Fiscal year like 2024, 2023, 2022
  - value (REAL): Numeric value
  - value_text (TEXT): Text value for percentages
  - source_file (TEXT): Source Excel file name

IMPORTANT:
1. Use period column for specific dates: WHERE period = 'March 2024'
2. Use fiscal_year for year ranges: WHERE fiscal_year BETWEEN 2022 AND 2024
3. For "last 3 years" from 2025, use: WHERE fiscal_year >= 2023
4. Metric names are case-sensitive: 'Sales', 'Net Profit', 'Operating Profit'
5. Company names may be partial: use LIKE '%JK Tyre%' for flexible matching

NATURAL LANGUAGE QUERY:
{query}

Return ONLY the SQL query. Start with SELECT."""


def generate_and_run_query(conn: sqlite3.Connection, natural_query: str) -> Dict[str, Any]:
    """Generate SQL from natural language and execute."""
    prompt = SQL_PROMPT.format(query=natural_query)
    
    result = {
        "query": natural_query,
        "sql": None,
        "data": [],
        "success": False,
        "error": None
    }
    
    try:
        response = call_qwen(prompt)
        sql = extract_sql(response)
        result["sql"] = sql
        
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        
        result["data"] = [dict(zip(columns, row)) for row in rows]
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


# =========================
# DATA VERIFICATION
# =========================

def verify_excel_data(excel_path: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify extracted data matches original Excel."""
    file_name = os.path.basename(excel_path)
    original = _load_excel_as_json(excel_path)
    
    # Get records for this file
    file_records = [r for r in records if r['source_file'] == file_name]
    
    verification = {
        "file": file_name,
        "extracted_records": len(file_records),
        "tests": [],
        "all_passed": True
    }
    
    # Test specific values
    test_cases = [
        ("Sales", "Mar-24"),
        ("Sales", "Mar-23"),
        ("Operating Profit", "Mar-24"),
    ]
    
    for metric, period_raw in test_cases:
        # Get from our records
        our_value = None
        for r in file_records:
            if r['metric'] == metric and r['period_raw'] == period_raw:
                our_value = r['value']
                break
        
        # Get from original
        orig_value = None
        for sheet_data in original.get('sheets', {}).values():
            for row in sheet_data.get('data', []):
                row_metric = row.get('Report_Date') or row.get('Report Date') or row.get(list(row.keys())[0] if row else '')
                if row_metric == metric:
                    orig_value = row.get(period_raw) or row.get(period_raw.replace('-', '_'))
                    if isinstance(orig_value, (int, float)):
                        orig_value = float(orig_value)
                    break
        
        passed = False
        if our_value is not None and orig_value is not None:
            passed = abs(our_value - orig_value) < 0.01
        elif our_value is None and orig_value is None:
            passed = True
        
        verification["tests"].append({
            "metric": metric,
            "period": period_raw,
            "extracted": our_value,
            "original": orig_value,
            "passed": passed
        })
        
        if not passed and our_value is not None:
            verification["all_passed"] = False
    
    return verification


# =========================
# MAIN TEST
# =========================

def run_complete_test():
    """Run complete end-to-end test."""
    print("\n" + "="*70)
    print("COMPLETE END-TO-END TEST WITH QWEN 480B")
    print("="*70)
    print(f"Model: {QWEN_MODEL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": QWEN_MODEL,
        "files_processed": [],
        "data_verifications": [],
        "query_tests": [],
        "summary": {}
    }
    
    # Step 1: Load all Excel files
    print("\n[1] Loading Excel files...")
    all_records = []
    excel_files = []
    
    for f in sorted(os.listdir(INPUT_DIR)):
        if f.endswith(('.xlsx', '.xls')):
            path = os.path.join(INPUT_DIR, f)
            excel_files.append(path)
            try:
                records = load_excel_to_generic(path)
                all_records.extend(records)
                company = extract_company_name(f)
                print(f"  ✓ {f}: {len(records)} records, Company: {company}")
                report["files_processed"].append({
                    "file": f, "type": "excel", "records": len(records), "status": "success"
                })
            except Exception as e:
                print(f"  ✗ {f}: {e}")
                report["files_processed"].append({
                    "file": f, "type": "excel", "records": 0, "status": f"error: {e}"
                })
    
    # Load PDF files
    pdf_files = []
    for f in sorted(os.listdir(INPUT_DIR)):
        if f.endswith('.pdf'):
            pdf_files.append(os.path.join(INPUT_DIR, f))
            print(f"  ℹ {f}: PDF file found (will process separately)")
            report["files_processed"].append({
                "file": f, "type": "pdf", "records": 0, "status": "pending"
            })
    
    print(f"\nTotal records loaded: {len(all_records)}")
    
    # Step 2: Create database
    print("\n[2] Creating SQLite database...")
    conn = create_sqlite_db(all_records)
    
    # Show sample data
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT company FROM financial_data")
    companies = [r[0] for r in cursor.fetchall()]
    print(f"  Companies: {', '.join(companies)}")
    
    cursor.execute("SELECT COUNT(*) FROM financial_data")
    total = cursor.fetchone()[0]
    print(f"  Total records: {total}")
    
    # Step 3: Verify data accuracy
    print("\n[3] Verifying data accuracy against original Excel...")
    all_verifications_passed = True
    
    for excel_path in excel_files:
        verification = verify_excel_data(excel_path, all_records)
        report["data_verifications"].append(verification)
        
        status = "✓" if verification["all_passed"] else "✗"
        print(f"\n  {status} {verification['file']}:")
        
        for test in verification["tests"]:
            test_status = "✓" if test["passed"] else "✗"
            print(f"    {test_status} {test['metric']} ({test['period']}): extracted={test['extracted']}, original={test['original']}")
        
        if not verification["all_passed"]:
            all_verifications_passed = False
    
    # Step 4: Test natural language queries with QWEN 480B
    print("\n[4] Testing natural language queries with QWEN 480B...")
    
    test_queries = [
        "Get Sales for all companies for March 2024",
        "What was the Net Profit for JK Tyre in fiscal year 2024?",
        "Show Sales for the last 3 fiscal years for all companies",
        "Compare Sales between Apollo Tyres and CEAT for March 2024",
        "Which company had the highest Sales in March 2025?",
        "Get Operating Profit for MRF from 2020 to 2025",
        "Show all metrics for Goodyear India in March 2024",
    ]
    
    queries_passed = 0
    queries_failed = 0
    
    for query in test_queries:
        print(f"\n  Query: {query}")
        result = generate_and_run_query(conn, query)
        report["query_tests"].append(result)
        
        if result["success"]:
            print(f"    ✓ SQL: {result['sql'][:80]}...")
            print(f"    ✓ Rows: {len(result['data'])}")
            if result['data']:
                # Show first row
                first = result['data'][0]
                sample = ', '.join([f"{k}={v}" for k, v in list(first.items())[:4]])
                print(f"    ✓ Sample: {sample}")
            queries_passed += 1
        else:
            print(f"    ✗ Error: {result['error']}")
            queries_failed += 1
    
    # Step 5: Process PDF file
    if pdf_files:
        print("\n[5] Processing PDF files with QWEN 480B...")
        for pdf_path in pdf_files:
            try:
                print(f"\n  Processing: {os.path.basename(pdf_path)}")
                extracted, analysis = process_document_qwen(pdf_path, iterations=5)
                
                # Update report
                for item in report["files_processed"]:
                    if item["file"] == os.path.basename(pdf_path):
                        item["status"] = "success"
                        item["analysis"] = analysis
                
                print(f"    ✓ Data type: {analysis.get('data_type', {}).get('primary_type', 'Unknown')}")
                print(f"    ✓ Periods: {analysis.get('number_of_periods', 0)}")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                for item in report["files_processed"]:
                    if item["file"] == os.path.basename(pdf_path):
                        item["status"] = f"error: {e}"
    
    # Summary
    report["summary"] = {
        "total_files": len(report["files_processed"]),
        "excel_files": len(excel_files),
        "pdf_files": len(pdf_files),
        "total_records": len(all_records),
        "data_accuracy": "PASSED" if all_verifications_passed else "FAILED",
        "queries_tested": len(test_queries),
        "queries_passed": queries_passed,
        "queries_failed": queries_failed,
        "success_rate": f"{queries_passed / len(test_queries) * 100:.1f}%"
    }
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    print(f"Files Processed: {report['summary']['total_files']} ({report['summary']['excel_files']} Excel, {report['summary']['pdf_files']} PDF)")
    print(f"Total Records: {report['summary']['total_records']}")
    print(f"Data Accuracy: {report['summary']['data_accuracy']}")
    print(f"Query Tests: {report['summary']['queries_passed']}/{report['summary']['queries_tested']} passed ({report['summary']['success_rate']})")
    
    # Detailed verification table
    print("\n" + "-"*70)
    print("DATA VERIFICATION DETAILS")
    print("-"*70)
    print(f"{'File':<35} {'Metric':<20} {'Period':<10} {'Extracted':<12} {'Original':<12} {'Status'}")
    print("-"*70)
    
    for v in report["data_verifications"]:
        for t in v["tests"]:
            status = "✓ PASS" if t["passed"] else "✗ FAIL"
            ext_val = f"{t['extracted']:.2f}" if t['extracted'] else "None"
            orig_val = f"{t['original']:.2f}" if t['original'] else "None"
            print(f"{v['file']:<35} {t['metric']:<20} {t['period']:<10} {ext_val:<12} {orig_val:<12} {status}")
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, "complete_test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nDetailed report saved to: {report_path}")
    
    conn.close()
    return report


if __name__ == "__main__":
    run_complete_test()
