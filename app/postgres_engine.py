"""
PostgreSQL-based Query Engine with Generic Data Structure

This module:
1. Stores financial data in a normalized PostgreSQL structure
2. Uses JSONB for flexible raw data storage
3. Converts natural language to SQL using QWEN 480B
4. Works with generic period names (March 2024, not Mar_24)
"""

import json
import os
import re
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from decimal import Decimal

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import requests

# =========================
# CONFIG
# =========================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
QWEN_MODEL = "qwen3-coder:480b-cloud"

# PostgreSQL connection
PG_CONFIG = {
    "host": "127.0.0.1",
    "port": 5432,
    "database": "llmdb",
    "user": "llm_user",
    "password": "Matrix@2026"
}


# =========================
# DATABASE SETUP
# =========================

def get_connection():
    """Get PostgreSQL connection."""
    return psycopg2.connect(**PG_CONFIG)


def setup_database():
    """Create tables for generic financial data storage."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Drop existing tables for clean setup
    cursor.execute("DROP TABLE IF EXISTS financial_data CASCADE")
    cursor.execute("DROP TABLE IF EXISTS source_files CASCADE")
    
    # Source files table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS source_files (
            id SERIAL PRIMARY KEY,
            file_name VARCHAR(255) NOT NULL,
            file_type VARCHAR(50) NOT NULL,
            company_name VARCHAR(255),
            data_type VARCHAR(100),
            loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            raw_json JSONB,
            UNIQUE(file_name)
        )
    """)
    
    # Normalized financial data table
    # Generic structure: metric + period + value
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS financial_data (
            id SERIAL PRIMARY KEY,
            source_file_id INTEGER REFERENCES source_files(id),
            company_name VARCHAR(255) NOT NULL,
            metric_name VARCHAR(255) NOT NULL,
            period_label VARCHAR(50) NOT NULL,
            period_date DATE,
            fiscal_year INTEGER,
            value DECIMAL(20, 4),
            value_text VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes for fast queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fd_company ON financial_data(company_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fd_metric ON financial_data(metric_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fd_period ON financial_data(period_label)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_fd_fiscal_year ON financial_data(fiscal_year)")
    
    conn.commit()
    cursor.close()
    conn.close()
    print("✓ Database tables created successfully")


def extract_company_name(filename: str) -> str:
    """Extract company name from filename."""
    name = filename.replace('.xlsx', '').replace('.xls', '').replace('.pdf', '')
    name = re.sub(r'_?(Target|Peer \d+)$', '', name).strip()
    return name


def _analyze_data_type(data: Dict[str, Any]) -> str:
    """Use LLM to identify the document type from extracted data."""
    # Get sample of metrics/columns to analyze
    sample_metrics = []
    for sheet_name, sheet_data in data.get('sheets', {}).items():
        for row in sheet_data.get('data', [])[:10]:
            metric = row.get('Report Date') or row.get('Report_Date') or row.get(list(row.keys())[0] if row else '')
            if metric and str(metric).strip():
                sample_metrics.append(str(metric))
    
    if not sample_metrics:
        return "Unknown"
    
    prompt = f"""Analyze these financial metrics and identify the document type.

Metrics found: {', '.join(sample_metrics[:20])}

What type of financial document is this? Choose ONE from:
- Profit and Loss Statement
- Balance Sheet
- Cash Flow Statement
- Financial Ratios
- Revenue Breakdown
- Unknown

Return ONLY the document type name, nothing else."""

    try:
        response = _call_qwen(prompt, timeout=60)
        # Clean response
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        # Extract just the type
        for doc_type in ["Profit and Loss Statement", "Balance Sheet", "Cash Flow Statement", 
                         "Financial Ratios", "Revenue Breakdown"]:
            if doc_type.lower() in response.lower():
                return doc_type
        return response.strip()[:100]
    except Exception as e:
        return "Unknown"


def parse_period_label(period_str: str) -> Tuple[str, Optional[datetime], Optional[int]]:
    """
    Parse period string like 'Mar-16' into normalized label, date, and fiscal year.
    Returns (normalized_label, date, fiscal_year)
    """
    period_str = str(period_str).strip()
    
    # Handle Mar-YY format (e.g., Mar-16, Mar-24)
    match = re.match(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[_-](\d{2})', period_str, re.IGNORECASE)
    if match:
        month_str = match.group(1).capitalize()
        year_short = int(match.group(2))
        year_full = 2000 + year_short if year_short < 50 else 1900 + year_short
        
        month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        month = month_map.get(month_str, 3)
        
        # Normalize to full format
        month_full = {'Jan': 'January', 'Feb': 'February', 'Mar': 'March', 'Apr': 'April',
                      'May': 'May', 'Jun': 'June', 'Jul': 'July', 'Aug': 'August',
                      'Sep': 'September', 'Oct': 'October', 'Nov': 'November', 'Dec': 'December'}
        
        normalized = f"{month_full[month_str]} {year_full}"
        date = datetime(year_full, month, 1)
        
        # Fiscal year (Indian fiscal year ends in March)
        fiscal_year = year_full if month <= 3 else year_full + 1
        
        return normalized, date, fiscal_year
    
    # Handle full year format
    if period_str.isdigit() and len(period_str) == 4:
        year = int(period_str)
        return f"FY {year}", datetime(year, 3, 31), year
    
    return period_str, None, None


def load_excel_to_postgres(excel_path: str) -> int:
    """
    Load Excel file into PostgreSQL with normalized structure.
    Returns source_file_id.
    """
    from app.qwen_pipeline import _load_excel_as_json
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Extract data
    data = _load_excel_as_json(excel_path)
    file_name = os.path.basename(excel_path)
    
    # Extract company name from filename
    company_name = file_name.replace('.xlsx', '').replace('.xls', '')
    company_name = re.sub(r'_?(Target|Peer \d+)$', '', company_name).strip()
    
    # Use LLM to identify document type
    data_type = _analyze_data_type(data)
    
    # Insert source file
    cursor.execute("""
        INSERT INTO source_files (file_name, file_type, company_name, data_type, raw_json)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (file_name) DO UPDATE SET raw_json = EXCLUDED.raw_json, data_type = EXCLUDED.data_type, loaded_at = CURRENT_TIMESTAMP
        RETURNING id
    """, (file_name, 'excel', company_name, data_type, Json(data)))
    
    source_file_id = cursor.fetchone()[0]
    
    # Delete existing financial data for this file
    cursor.execute("DELETE FROM financial_data WHERE source_file_id = %s", (source_file_id,))
    
    # Process each sheet
    rows_inserted = 0
    for sheet_name, sheet_data in data.get('sheets', {}).items():
        columns = sheet_data.get('columns', [])
        rows = sheet_data.get('data', [])
        
        # Find the metric column (usually first column or 'Report Date')
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
        
        if not period_cols:
            # Try to find period columns
            for col in columns:
                if col != metric_col and not str(col).startswith('Column_'):
                    period_cols.append(col)
        
        # Insert normalized data
        for row in rows:
            metric_name = row.get(metric_col)
            if not metric_name or str(metric_name).strip() == '':
                continue
            
            metric_name = str(metric_name).strip()
            
            for period_col in period_cols:
                value = row.get(period_col)
                if value is None:
                    continue
                
                period_label, period_date, fiscal_year = parse_period_label(str(period_col))
                
                # Handle numeric vs text values
                value_numeric = None
                value_text = None
                
                if isinstance(value, (int, float)):
                    value_numeric = value
                else:
                    value_str = str(value).strip()
                    if value_str.endswith('%'):
                        value_text = value_str
                        try:
                            value_numeric = float(value_str.rstrip('%'))
                        except:
                            pass
                    else:
                        try:
                            value_numeric = float(value_str)
                        except:
                            value_text = value_str
                
                cursor.execute("""
                    INSERT INTO financial_data 
                    (source_file_id, company_name, metric_name, period_label, period_date, fiscal_year, value, value_text)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (source_file_id, company_name, metric_name, period_label, period_date, fiscal_year, value_numeric, value_text))
                rows_inserted += 1
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"✓ Loaded {file_name}: {rows_inserted} data points")
    return source_file_id


def load_pdf_to_postgres(pdf_path: str) -> int:
    """
    Load PDF file into PostgreSQL using QWEN extraction.
    Returns source_file_id.
    """
    from app.qwen_pipeline import process_document_qwen
    
    conn = get_connection()
    cursor = conn.cursor()
    
    file_name = os.path.basename(pdf_path)
    company_name = file_name.replace('.pdf', '').replace('.PDF', '').strip()
    
    # Extract data using QWEN pipeline
    print(f"Extracting data from PDF: {file_name}")
    extracted_data, analysis = process_document_qwen(pdf_path, iterations=5)
    
    # Insert source file with raw JSON
    cursor.execute("""
        INSERT INTO source_files (file_name, file_type, company_name, data_type, raw_json)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (file_name) DO UPDATE SET raw_json = EXCLUDED.raw_json, loaded_at = CURRENT_TIMESTAMP
        RETURNING id
    """, (file_name, 'pdf', company_name, 
          analysis.get('data_type', {}).get('primary_type', 'Unknown'),
          Json({'extracted_data': extracted_data, 'analysis': analysis})))
    
    source_file_id = cursor.fetchone()[0]
    
    # Delete existing financial data for this file
    cursor.execute("DELETE FROM financial_data WHERE source_file_id = %s", (source_file_id,))
    
    # Process extracted data - handle various structures
    rows_inserted = 0
    
    def process_item(key, value, parent_key=''):
        nonlocal rows_inserted
        full_key = f"{parent_key}.{key}" if parent_key else key
        
        if isinstance(value, dict):
            # Check if it's a period-value structure
            if 'period' in value and 'value' in value:
                period_label, period_date, fiscal_year = parse_period_label(value['period'])
                val = value['value']
                value_numeric = val if isinstance(val, (int, float)) else None
                value_text = str(val) if not isinstance(val, (int, float)) else None
                
                cursor.execute("""
                    INSERT INTO financial_data 
                    (source_file_id, company_name, metric_name, period_label, period_date, fiscal_year, value, value_text)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (source_file_id, company_name, parent_key or key, period_label, period_date, fiscal_year, value_numeric, value_text))
                rows_inserted += 1
            else:
                for k, v in value.items():
                    process_item(k, v, full_key)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    process_item(key, item, parent_key)
    
    if isinstance(extracted_data, dict):
        for key, value in extracted_data.items():
            process_item(key, value)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"✓ Loaded {file_name}: {rows_inserted} data points")
    return source_file_id


# =========================
# QUERY FUNCTIONS
# =========================

def get_schema_description() -> str:
    """Get human-readable schema description for LLM."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get companies
    cursor.execute("SELECT DISTINCT company_name FROM financial_data ORDER BY company_name")
    companies = [r['company_name'] for r in cursor.fetchall()]
    
    # Get metrics
    cursor.execute("SELECT DISTINCT metric_name FROM financial_data ORDER BY metric_name LIMIT 50")
    metrics = [r['metric_name'] for r in cursor.fetchall()]
    
    # Get periods
    cursor.execute("SELECT DISTINCT period_label, fiscal_year FROM financial_data ORDER BY fiscal_year, period_label")
    periods = [(r['period_label'], r['fiscal_year']) for r in cursor.fetchall()]
    
    cursor.close()
    conn.close()
    
    schema = f"""
DATABASE SCHEMA:

Table: financial_data
Columns:
  - company_name (VARCHAR): Company name, e.g., {', '.join(companies[:5])}
  - metric_name (VARCHAR): Financial metric, e.g., {', '.join(metrics[:10])}...
  - period_label (VARCHAR): Period in format "March 2024", "March 2023", etc.
  - fiscal_year (INTEGER): Fiscal year, e.g., 2024, 2023, 2022
  - value (DECIMAL): Numeric value
  - value_text (VARCHAR): Text value (for percentages like "27%")

Available Companies: {', '.join(companies)}
Available Periods: {', '.join([f"{p[0]} (FY{p[1]})" for p in periods[:10]])}...
Sample Metrics: {', '.join(metrics[:15])}...

IMPORTANT QUERY RULES:
1. Use period_label for specific periods: WHERE period_label = 'March 2024'
2. Use fiscal_year for year-based queries: WHERE fiscal_year = 2024
3. For "last 3 years", use: WHERE fiscal_year >= 2023 AND fiscal_year <= 2025
4. Metric names are case-sensitive: 'Sales', 'Net Profit', 'Operating Profit'
5. For comparisons, use GROUP BY and aggregations
"""
    return schema


def _call_qwen(prompt: str, timeout: int = 300) -> str:
    """Call QWEN 480B model via Ollama."""
    payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": False,
    }
    
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/v1/chat/completions",
            json=payload,
            timeout=timeout
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"QWEN call failed: {e}")


def _extract_sql_from_response(response: str) -> str:
    """Extract SQL query from LLM response."""
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


SQL_PROMPT_TEMPLATE = """You are a SQL expert. Convert this natural language query to PostgreSQL SQL.

{schema}

NATURAL LANGUAGE QUERY:
{query}

Generate a PostgreSQL SELECT query. Use the financial_data table.
Return ONLY the SQL query, no explanation. Start with SELECT."""


def generate_and_execute_query(natural_query: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Generate SQL from natural language and execute against PostgreSQL.
    """
    schema = get_schema_description()
    prompt = SQL_PROMPT_TEMPLATE.format(schema=schema, query=natural_query)
    
    result = {
        "natural_query": natural_query,
        "sql_query": None,
        "columns": [],
        "data": [],
        "success": False,
        "error": None,
        "attempts": []
    }
    
    for attempt in range(max_retries):
        try:
            # Generate SQL
            response = _call_qwen(prompt)
            sql = _extract_sql_from_response(response)
            result["attempts"].append({"attempt": attempt + 1, "sql": sql})
            
            # Execute query
            conn = get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(sql)
            rows = cursor.fetchall()
            
            result["sql_query"] = sql
            result["columns"] = list(rows[0].keys()) if rows else []
            result["data"] = [dict(r) for r in rows]
            result["success"] = True
            
            cursor.close()
            conn.close()
            return result
            
        except Exception as e:
            result["attempts"][-1]["error"] = str(e)
            
            # Add error context to prompt for retry
            prompt = SQL_PROMPT_TEMPLATE.format(schema=schema, query=natural_query)
            prompt += f"\n\nPrevious attempt failed with error: {e}\nPlease fix the SQL."
    
    result["error"] = "All attempts failed"
    return result


# =========================
# VERIFICATION
# =========================

def verify_data_against_excel(excel_path: str) -> Dict[str, Any]:
    """
    Verify PostgreSQL data matches original Excel data.
    Returns verification report.
    """
    from app.qwen_pipeline import _load_excel_as_json
    
    file_name = os.path.basename(excel_path)
    original_data = _load_excel_as_json(excel_path)
    
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get data from PostgreSQL for this file
    cursor.execute("""
        SELECT metric_name, period_label, value, value_text
        FROM financial_data fd
        JOIN source_files sf ON fd.source_file_id = sf.id
        WHERE sf.file_name = %s
        ORDER BY metric_name, period_label
    """, (file_name,))
    
    pg_data = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # Build verification report
    report = {
        "file_name": file_name,
        "postgres_row_count": len(pg_data),
        "sample_verifications": [],
        "all_match": True
    }
    
    # Sample verifications - test specific values
    test_cases = [
        ("Sales", "March 2024", "Mar-24"),
        ("Sales", "March 2023", "Mar-23"),
        ("Operating Profit", "March 2024", "Mar-24"),
    ]
    
    # Get company name for this file
    company_name = extract_company_name(file_name)
    
    for metric, period_label, period_col in test_cases:
        # Get from PostgreSQL - query directly for exact match
        conn2 = get_connection()
        cur2 = conn2.cursor()
        cur2.execute("""
            SELECT value FROM financial_data 
            WHERE company_name = %s AND metric_name = %s AND period_label = %s
            ORDER BY value DESC LIMIT 1
        """, (company_name, metric, period_label))
        row = cur2.fetchone()
        pg_value = float(row[0]) if row and row[0] else None
        cur2.close()
        conn2.close()
        
        # Get from original Excel - look in Data Sheet first
        orig_value = None
        for sheet_name in ['Data Sheet', 'Data_Sheet', 'Profit & Loss']:
            sheet_data = original_data.get('sheets', {}).get(sheet_name, {})
            if not sheet_data:
                continue
            for row in sheet_data.get('data', []):
                row_metric = row.get('Report Date') or row.get('Report_Date')
                if row_metric == metric:
                    val = row.get(period_col)
                    if val is not None and isinstance(val, (int, float)):
                        orig_value = float(val)
                        break
            if orig_value is not None:
                break
        
        # Check match
        match = False
        if pg_value is not None and orig_value is not None:
            match = abs(pg_value - orig_value) < 0.01
        elif pg_value is None and orig_value is None:
            match = True
        
        report["sample_verifications"].append({
            "metric": metric,
            "period": period_label,
            "postgres_value": pg_value,
            "original_value": orig_value,
            "match": match
        })
        
        if not match and pg_value is not None:
            report["all_match"] = False
    
    return report


# =========================
# MAIN TEST RUNNER
# =========================

def run_full_test(input_dir: str = "input") -> Dict[str, Any]:
    """
    Run complete end-to-end test on all files.
    Returns detailed report.
    """
    print("\n" + "="*70)
    print("COMPLETE END-TO-END TEST WITH QWEN 480B")
    print("="*70)
    
    # Setup database
    print("\n[1] Setting up PostgreSQL database...")
    setup_database()
    
    # Load all files
    print("\n[2] Loading all files into PostgreSQL...")
    files_loaded = []
    
    for f in os.listdir(input_dir):
        path = os.path.join(input_dir, f)
        if f.endswith(('.xlsx', '.xls')):
            try:
                load_excel_to_postgres(path)
                files_loaded.append({"file": f, "type": "excel", "status": "loaded"})
            except Exception as e:
                files_loaded.append({"file": f, "type": "excel", "status": f"error: {e}"})
        elif f.endswith('.pdf'):
            try:
                load_pdf_to_postgres(path)
                files_loaded.append({"file": f, "type": "pdf", "status": "loaded"})
            except Exception as e:
                files_loaded.append({"file": f, "type": "pdf", "status": f"error: {e}"})
    
    # Verify data
    print("\n[3] Verifying data accuracy...")
    verifications = []
    for f in os.listdir(input_dir):
        if f.endswith(('.xlsx', '.xls')):
            path = os.path.join(input_dir, f)
            verification = verify_data_against_excel(path)
            verifications.append(verification)
            status = "✓" if verification["all_match"] else "✗"
            print(f"  {status} {f}: {verification['postgres_row_count']} rows")
    
    # Test queries
    print("\n[4] Testing natural language queries with QWEN 480B...")
    test_queries = [
        "Get Sales for all companies for March 2024",
        "What was the Net Profit for JK Tyre in fiscal year 2024?",
        "Show me Sales and Operating Profit for MRF for the last 3 years",
        "Compare Sales between Apollo Tyres and CEAT for March 2024",
        "Which company had the highest Sales in March 2025?",
        "Get the total of Sales for all companies in fiscal year 2024",
        "Show Operating Profit trend for Goodyear India from 2020 to 2025",
    ]
    
    query_results = []
    for query in test_queries:
        print(f"\n  Query: {query}")
        result = generate_and_execute_query(query)
        query_results.append(result)
        
        if result["success"]:
            print(f"    ✓ SQL: {result['sql_query'][:80]}...")
            print(f"    ✓ Rows: {len(result['data'])}")
            if result['data']:
                print(f"    ✓ Sample: {result['data'][0]}")
        else:
            print(f"    ✗ Error: {result['error']}")
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": QWEN_MODEL,
        "files_loaded": files_loaded,
        "data_verifications": verifications,
        "query_tests": query_results,
        "summary": {
            "total_files": len(files_loaded),
            "files_success": sum(1 for f in files_loaded if f["status"] == "loaded"),
            "total_queries": len(query_results),
            "queries_success": sum(1 for q in query_results if q["success"]),
            "data_accuracy": sum(1 for v in verifications if v["all_match"]) / len(verifications) * 100 if verifications else 0
        }
    }
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    print(f"Files Loaded: {report['summary']['files_success']}/{report['summary']['total_files']}")
    print(f"Queries Passed: {report['summary']['queries_success']}/{report['summary']['total_queries']}")
    print(f"Data Accuracy: {report['summary']['data_accuracy']:.1f}%")
    
    return report


if __name__ == "__main__":
    report = run_full_test()
    
    # Save report
    with open("processed/full_test_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: processed/full_test_report.json")
