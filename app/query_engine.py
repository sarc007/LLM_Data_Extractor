"""
Natural Language to SQL Query Engine

This module converts natural language queries to SQL using LLM,
executes them against extracted financial data, and validates results.

Supports:
- gemma3:4b (primary, faster)
- qwen3-coder:480b-cloud (fallback, more accurate)
"""

import json
import os
import re
import sqlite3
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import requests

# =========================
# CONFIG
# =========================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
PRIMARY_MODEL = "gemma3:4b"
FALLBACK_MODEL = "qwen3-coder:480b-cloud"


# =========================
# DATABASE FUNCTIONS
# =========================

def create_database_from_json(extracted_data: Dict[str, Any], db_path: str = ":memory:") -> sqlite3.Connection:
    """
    Create SQLite database from extracted JSON data.
    Returns a connection to the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get sheets data
    sheets = extracted_data.get("sheets", {})
    
    for sheet_name, sheet_data in sheets.items():
        # Clean table name for SQL
        table_name = _clean_table_name(sheet_name)
        columns = sheet_data.get("columns", [])
        rows = sheet_data.get("data", [])
        
        if not columns or not rows:
            continue
        
        # Determine column types from data
        col_types = _infer_column_types(columns, rows)
        
        # Create table
        col_defs = ", ".join([f'"{_clean_column_name(col)}" {col_types[col]}' for col in columns])
        create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs})'
        cursor.execute(create_sql)
        
        # Insert data
        placeholders = ", ".join(["?" for _ in columns])
        clean_cols = [_clean_column_name(c) for c in columns]
        insert_sql = f'INSERT INTO "{table_name}" ({", ".join([f"{c}" for c in clean_cols])}) VALUES ({placeholders})'
        
        for row in rows:
            values = [row.get(col) for col in columns]
            cursor.execute(insert_sql, values)
    
    conn.commit()
    return conn


def _clean_table_name(name: str) -> str:
    """Clean table name for SQL."""
    # Replace spaces and special chars with underscores
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    return clean.strip('_')


def _clean_column_name(name: str) -> str:
    """Clean column name for SQL."""
    # Handle period columns like Mar-16
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', str(name))
    # If starts with number, prefix with underscore
    if clean and clean[0].isdigit():
        clean = '_' + clean
    return clean.strip('_')


def _infer_column_types(columns: List[str], rows: List[Dict]) -> Dict[str, str]:
    """Infer SQL column types from data."""
    col_types = {}
    for col in columns:
        # Check first few non-null values
        sample_values = [row.get(col) for row in rows[:10] if row.get(col) is not None]
        
        if not sample_values:
            col_types[col] = "TEXT"
        elif all(isinstance(v, (int, float)) for v in sample_values):
            col_types[col] = "REAL"
        else:
            col_types[col] = "TEXT"
    
    return col_types


def get_schema_description(conn: sqlite3.Connection) -> str:
    """Get human-readable schema description for LLM."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    schema_parts = []
    for (table_name,) in tables:
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()
        
        col_info = []
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            col_info.append(f"  - {col_name} ({col_type})")
        
        # Get sample data
        cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 3')
        sample_rows = cursor.fetchall()
        
        schema_parts.append(f"Table: {table_name}")
        schema_parts.append("Columns:")
        schema_parts.extend(col_info)
        if sample_rows:
            schema_parts.append("Sample data (first 3 rows):")
            col_names = [c[1] for c in columns]
            for row in sample_rows:
                row_str = ", ".join([f"{col_names[i]}={row[i]}" for i in range(min(5, len(row)))])
                schema_parts.append(f"  {row_str}...")
        schema_parts.append("")
    
    return "\n".join(schema_parts)


# =========================
# LLM FUNCTIONS
# =========================

def _call_llm(prompt: str, model: str, timeout: int = 120) -> str:
    """Call LLM via Ollama."""
    payload = {
        "model": model,
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
        raise RuntimeError(f"LLM call failed ({model}): {e}")


def _extract_sql_from_response(response: str) -> str:
    """Extract SQL query from LLM response."""
    # Remove thinking tags
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    
    # Try to find SQL in code blocks
    sql_match = re.search(r"```sql\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    if sql_match:
        return sql_match.group(1).strip()
    
    # Try generic code block
    code_match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # Look for SELECT statement
    select_match = re.search(r"(SELECT\s+.*?;)", response, re.DOTALL | re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip()
    
    # Return cleaned response
    return response.strip()


# =========================
# QUERY GENERATION
# =========================

SQL_GENERATION_PROMPT = """You are a SQL expert. Convert the natural language query to SQLite SQL.

DATABASE SCHEMA:
{schema}

IMPORTANT RULES:
1. The main data table is usually "Data_Sheet" with columns like:
   - Report_Date: Contains row labels like "Sales", "Net Profit", "EPS", etc.
   - Period columns: Mar_16, Mar_17, Mar_18, ... Mar_25 (financial year ending March)
2. To get Sales for Mar-24: SELECT Mar_24 FROM Data_Sheet WHERE Report_Date = 'Sales'
3. To compare multiple years: SELECT Report_Date, Mar_22, Mar_23, Mar_24 FROM Data_Sheet WHERE Report_Date IN ('Sales', 'Net Profit')
4. For "last 3 years" from Mar-25, use: Mar_23, Mar_24, Mar_25
5. Column names with hyphens are converted to underscores: Mar-16 becomes Mar_16
6. Use double quotes for column/table names if they contain special characters

NATURAL LANGUAGE QUERY:
{query}

Return ONLY the SQL query, no explanation. Start with SELECT."""


def generate_sql(query: str, schema: str, model: str = PRIMARY_MODEL) -> Tuple[str, str]:
    """
    Generate SQL from natural language query.
    Returns (sql_query, model_used)
    """
    prompt = SQL_GENERATION_PROMPT.format(schema=schema, query=query)
    
    try:
        response = _call_llm(prompt, model)
        sql = _extract_sql_from_response(response)
        return sql, model
    except Exception as e:
        raise RuntimeError(f"SQL generation failed: {e}")


def execute_query(conn: sqlite3.Connection, sql: str) -> Tuple[List[str], List[Tuple]]:
    """
    Execute SQL query and return (column_names, rows).
    """
    cursor = conn.cursor()
    cursor.execute(sql)
    columns = [desc[0] for desc in cursor.description] if cursor.description else []
    rows = cursor.fetchall()
    return columns, rows


# =========================
# QUERY WITH FALLBACK
# =========================

def query_with_fallback(
    conn: sqlite3.Connection,
    natural_query: str,
    schema: str,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Try to generate and execute SQL query with fallback to stronger model.
    Returns dict with query details and results.
    """
    result = {
        "natural_query": natural_query,
        "sql_query": None,
        "model_used": None,
        "columns": [],
        "data": [],
        "success": False,
        "error": None,
        "attempts": []
    }
    
    models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL]
    
    for model in models_to_try:
        for attempt in range(max_retries):
            attempt_info = {"model": model, "attempt": attempt + 1}
            
            try:
                # Generate SQL
                sql, _ = generate_sql(natural_query, schema, model)
                attempt_info["sql"] = sql
                
                # Execute query
                columns, rows = execute_query(conn, sql)
                
                result["sql_query"] = sql
                result["model_used"] = model
                result["columns"] = columns
                result["data"] = [dict(zip(columns, row)) for row in rows]
                result["success"] = True
                result["attempts"].append(attempt_info)
                
                return result
                
            except Exception as e:
                attempt_info["error"] = str(e)
                result["attempts"].append(attempt_info)
                
                # If SQL error, might need to fix the query
                if "no such column" in str(e).lower() or "no such table" in str(e).lower():
                    continue  # Try again with same model
                
        # Move to next model if all retries failed
    
    result["error"] = "All models and retries failed"
    return result


# =========================
# VALIDATION
# =========================

def validate_result(result: Dict[str, Any], expected_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Validate query result against expected data if provided.
    """
    validation = {
        "has_data": len(result.get("data", [])) > 0,
        "row_count": len(result.get("data", [])),
        "columns": result.get("columns", []),
    }
    
    if expected_data:
        validation["matches_expected"] = result.get("data") == expected_data
    
    return validation


# =========================
# TEST HARNESS
# =========================

def run_test_queries(
    extracted_data: Dict[str, Any],
    test_queries: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Run a list of test queries and return results with validation.
    
    test_queries format:
    [
        {"query": "Get sales for last 3 years", "expected_columns": ["Mar_23", "Mar_24", "Mar_25"]},
        ...
    ]
    """
    # Create database
    conn = create_database_from_json(extracted_data)
    schema = get_schema_description(conn)
    
    results = []
    for test in test_queries:
        query = test.get("query")
        print(f"\n[TEST] Query: {query}")
        
        result = query_with_fallback(conn, query, schema)
        result["test_info"] = test
        
        if result["success"]:
            print(f"  ✓ SQL: {result['sql_query']}")
            print(f"  ✓ Model: {result['model_used']}")
            print(f"  ✓ Rows: {len(result['data'])}")
            if result['data']:
                print(f"  ✓ Sample: {result['data'][0]}")
        else:
            print(f"  ✗ Error: {result['error']}")
            for att in result['attempts']:
                print(f"    - {att['model']} attempt {att['attempt']}: {att.get('error', 'OK')}")
        
        results.append(result)
    
    conn.close()
    return results


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import sys
    from app.qwen_pipeline import _load_excel_as_json
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.query_engine <excel_file> [query]")
        sys.exit(1)
    
    excel_path = sys.argv[1]
    data = _load_excel_as_json(excel_path)
    
    print(f"Loaded {excel_path}")
    print(f"Sheets: {list(data['sheets'].keys())}")
    
    # Create database
    conn = create_database_from_json(data)
    schema = get_schema_description(conn)
    print("\nSchema:")
    print(schema)
    
    if len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        print(f"\nQuery: {query}")
        result = query_with_fallback(conn, query, schema)
        print(f"SQL: {result['sql_query']}")
        print(f"Model: {result['model_used']}")
        print(f"Success: {result['success']}")
        if result['data']:
            print(f"Results: {result['data']}")
    
    conn.close()
