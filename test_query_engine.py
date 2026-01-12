"""
Test script for the query engine.
Tests natural language to SQL conversion with all Excel files.
"""

import sys
import os
sys.path.insert(0, '.')

from app.qwen_pipeline import _load_excel_as_json
from app.query_engine import (
    create_database_from_json, 
    get_schema_description, 
    query_with_fallback,
    execute_query
)

# Test queries to run against each file
TEST_QUERIES = [
    "Get Sales data for the last 3 years",
    "What was the Net Profit in Mar-24?",
    "Show me Sales and Expenses for Mar-22, Mar-23, Mar-24",
    "Which year had the highest EPS?",
    "Compare Operating Profit between Mar-20 and Mar-25",
    "Get all data for Sales row",
    "What is the total of Mar_24 column for Sales and Net Profit?",
]


def test_file(excel_path: str) -> dict:
    """Test all queries against a single Excel file."""
    print(f"\n{'='*60}")
    print(f"TESTING: {os.path.basename(excel_path)}")
    print('='*60)
    
    # Load data
    data = _load_excel_as_json(excel_path)
    print(f"Sheets: {list(data['sheets'].keys())}")
    
    # Create database
    conn = create_database_from_json(data)
    schema = get_schema_description(conn)
    
    # Show schema summary
    print("\nData_Sheet columns:", end=" ")
    if 'Data_Sheet' in [t.replace('_', ' ') for t in data['sheets'].keys()] or 'Data Sheet' in data['sheets']:
        sheet_key = 'Data Sheet' if 'Data Sheet' in data['sheets'] else 'Data_Sheet'
        if sheet_key in data['sheets']:
            cols = data['sheets'][sheet_key]['columns']
            print(cols[:5], "...", cols[-3:] if len(cols) > 5 else "")
    
    results = []
    passed = 0
    failed = 0
    
    for query in TEST_QUERIES:
        print(f"\n[Query] {query}")
        result = query_with_fallback(conn, query, schema)
        
        if result['success']:
            print(f"  ✓ Model: {result['model_used']}")
            print(f"  ✓ SQL: {result['sql_query']}")
            if result['data']:
                # Show first row of results
                first_row = result['data'][0]
                print(f"  ✓ Result ({len(result['data'])} rows): {first_row}")
            passed += 1
        else:
            print(f"  ✗ FAILED: {result['error']}")
            for att in result['attempts'][-2:]:  # Show last 2 attempts
                print(f"    - {att.get('model')}: {att.get('error', 'N/A')}")
                if att.get('sql'):
                    print(f"      SQL: {att['sql'][:100]}...")
            failed += 1
        
        results.append(result)
    
    conn.close()
    
    print(f"\n--- Summary for {os.path.basename(excel_path)} ---")
    print(f"Passed: {passed}/{len(TEST_QUERIES)}")
    print(f"Failed: {failed}/{len(TEST_QUERIES)}")
    
    return {
        'file': excel_path,
        'passed': passed,
        'failed': failed,
        'results': results
    }


def verify_data_accuracy(excel_path: str):
    """Verify that SQL query results match actual Excel data."""
    print(f"\n{'='*60}")
    print(f"VERIFYING DATA ACCURACY: {os.path.basename(excel_path)}")
    print('='*60)
    
    data = _load_excel_as_json(excel_path)
    conn = create_database_from_json(data)
    
    # Get actual values from Data_Sheet
    cursor = conn.cursor()
    
    # Test 1: Sales for Mar-24
    cursor.execute("SELECT Mar_24 FROM Data_Sheet WHERE Report_Date = 'Sales'")
    sql_result = cursor.fetchone()
    
    # Get from raw data
    raw_sales = None
    for sheet_data in data['sheets'].values():
        for row in sheet_data['data']:
            if row.get('Report_Date') == 'Sales' or row.get('Report Date') == 'Sales':
                raw_sales = row.get('Mar-24') or row.get('Mar_24')
                break
    
    print(f"\nTest: Sales for Mar-24")
    print(f"  SQL Result: {sql_result[0] if sql_result else 'None'}")
    print(f"  Raw Data: {raw_sales}")
    print(f"  Match: {'✓' if sql_result and abs(sql_result[0] - raw_sales) < 0.01 else '✗'}")
    
    # Test 2: Net Profit for Mar-23
    cursor.execute("SELECT Mar_23 FROM Data_Sheet WHERE Report_Date = 'Net Profit'")
    sql_result = cursor.fetchone()
    
    raw_np = None
    for sheet_data in data['sheets'].values():
        for row in sheet_data['data']:
            if row.get('Report_Date') == 'Net Profit' or row.get('Report Date') == 'Net Profit':
                raw_np = row.get('Mar-23') or row.get('Mar_23')
                break
    
    print(f"\nTest: Net Profit for Mar-23")
    print(f"  SQL Result: {sql_result[0] if sql_result else 'None'}")
    print(f"  Raw Data: {raw_np}")
    if sql_result and raw_np:
        print(f"  Match: {'✓' if abs(sql_result[0] - raw_np) < 0.01 else '✗'}")
    
    conn.close()


def main():
    """Run tests on all Excel files."""
    input_dir = 'input'
    excel_files = [f for f in os.listdir(input_dir) if f.endswith(('.xlsx', '.xls'))]
    
    print(f"Found {len(excel_files)} Excel files to test")
    
    all_results = []
    total_passed = 0
    total_failed = 0
    
    for excel_file in excel_files:
        excel_path = os.path.join(input_dir, excel_file)
        
        # First verify data accuracy
        verify_data_accuracy(excel_path)
        
        # Then test queries
        result = test_file(excel_path)
        all_results.append(result)
        total_passed += result['passed']
        total_failed += result['failed']
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print('='*60)
    print(f"Total files tested: {len(excel_files)}")
    print(f"Total queries: {len(TEST_QUERIES) * len(excel_files)}")
    print(f"Total passed: {total_passed}")
    print(f"Total failed: {total_failed}")
    print(f"Success rate: {total_passed / (total_passed + total_failed) * 100:.1f}%")


if __name__ == "__main__":
    main()
