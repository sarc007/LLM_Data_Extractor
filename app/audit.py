"""
Data Extraction Audit Module

Verifies that all data from source Excel/CSV files is present in extracted JSON.
Provides detailed reports on any data loss or discrepancies.
"""

import json
import os
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import numpy as np


@dataclass
class AuditResult:
    """Result of an audit comparison."""
    passed: bool
    source_file: str
    json_file: Optional[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Counts
    source_row_count: int = 0
    json_row_count: int = 0
    source_column_count: int = 0
    json_column_count: int = 0
    source_sheet_count: int = 0
    json_sheet_count: int = 0
    
    # Numeric verification
    source_numeric_sum: float = 0.0
    json_numeric_sum: float = 0.0
    source_numeric_count: int = 0
    json_numeric_count: int = 0
    
    # Discrepancies
    missing_sheets: List[str] = field(default_factory=list)
    missing_columns: Dict[str, List[str]] = field(default_factory=dict)
    row_count_mismatches: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    numeric_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "passed": self.passed,
            "source_file": self.source_file,
            "json_file": self.json_file,
            "timestamp": self.timestamp,
            "summary": {
                "source_rows": self.source_row_count,
                "json_rows": self.json_row_count,
                "row_match": self.source_row_count == self.json_row_count,
                "source_columns": self.source_column_count,
                "json_columns": self.json_column_count,
                "column_match": self.source_column_count == self.json_column_count,
                "source_sheets": self.source_sheet_count,
                "json_sheets": self.json_sheet_count,
                "sheet_match": self.source_sheet_count == self.json_sheet_count,
                "source_numeric_sum": round(self.source_numeric_sum, 2),
                "json_numeric_sum": round(self.json_numeric_sum, 2),
                "numeric_sum_match": abs(self.source_numeric_sum - self.json_numeric_sum) < 0.01,
                "source_numeric_count": self.source_numeric_count,
                "json_numeric_count": self.json_numeric_count,
            },
            "discrepancies": {
                "missing_sheets": self.missing_sheets,
                "missing_columns": self.missing_columns,
                "row_count_mismatches": self.row_count_mismatches,
                "numeric_mismatches": self.numeric_mismatches[:20],  # Limit for readability
            },
            "warnings": self.warnings,
        }
    
    def print_report(self):
        """Print a formatted audit report."""
        status = "[OK] PASSED" if self.passed else "[FAIL] FAILED"
        print(f"\n{'='*60}")
        print(f"AUDIT REPORT - {status}")
        print(f"{'='*60}")
        print(f"Source: {self.source_file}")
        print(f"JSON: {self.json_file}")
        print(f"Time: {self.timestamp}")
        
        print(f"\n{'-'*40}")
        print("SUMMARY")
        print(f"{'-'*40}")
        
        row_status = "[OK]" if self.source_row_count == self.json_row_count else "[FAIL]"
        print(f"  Rows:    {self.source_row_count:>6} source vs {self.json_row_count:>6} json {row_status}")
        
        col_status = "[OK]" if self.source_column_count == self.json_column_count else "[FAIL]"
        print(f"  Columns: {self.source_column_count:>6} source vs {self.json_column_count:>6} json {col_status}")
        
        sheet_status = "[OK]" if self.source_sheet_count == self.json_sheet_count else "[FAIL]"
        print(f"  Sheets:  {self.source_sheet_count:>6} source vs {self.json_sheet_count:>6} json {sheet_status}")
        
        numeric_diff = abs(self.source_numeric_sum - self.json_numeric_sum)
        numeric_status = "[OK]" if numeric_diff < 0.01 else "[FAIL]"
        print(f"\n  Numeric Sum: {self.source_numeric_sum:,.2f} source")
        print(f"               {self.json_numeric_sum:,.2f} json {numeric_status}")
        if numeric_diff >= 0.01:
            print(f"               Difference: {numeric_diff:,.2f}")
        
        print(f"  Numeric Values: {self.source_numeric_count} source vs {self.json_numeric_count} json")
        
        if self.missing_sheets:
            print(f"\n{'-'*40}")
            print("MISSING SHEETS")
            print(f"{'-'*40}")
            for sheet in self.missing_sheets:
                print(f"  [FAIL] {sheet}")
        
        if self.missing_columns:
            print(f"\n{'-'*40}")
            print("MISSING COLUMNS")
            print(f"{'-'*40}")
            for sheet, cols in self.missing_columns.items():
                print(f"  Sheet '{sheet}':")
                for col in cols[:10]:
                    print(f"    [FAIL] {col}")
                if len(cols) > 10:
                    print(f"    ... and {len(cols) - 10} more")
        
        if self.row_count_mismatches:
            print(f"\n{'-'*40}")
            print("ROW COUNT MISMATCHES")
            print(f"{'-'*40}")
            for sheet, (source, json_count) in self.row_count_mismatches.items():
                diff = source - json_count
                print(f"  Sheet '{sheet}': {source} source -> {json_count} json (missing {diff})")
        
        if self.numeric_mismatches:
            print(f"\n{'-'*40}")
            print(f"NUMERIC MISMATCHES (showing first 10 of {len(self.numeric_mismatches)})")
            print(f"{'-'*40}")
            for mismatch in self.numeric_mismatches[:10]:
                print(f"  {mismatch.get('location', 'unknown')}: "
                      f"{mismatch.get('source_value', 'N/A')} -> {mismatch.get('json_value', 'N/A')}")
        
        if self.warnings:
            print(f"\n{'-'*40}")
            print("WARNINGS")
            print(f"{'-'*40}")
            for warning in self.warnings:
                print(f"  [WARNING]  {warning}")
        
        print(f"\n{'='*60}\n")


def _extract_numeric_from_value(val: Any) -> Optional[float]:
    """Extract numeric value from various formats."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        # Handle percentage strings
        val = val.strip()
        if val.endswith('%'):
            try:
                return float(val[:-1])
            except ValueError:
                return None
        # Handle numeric strings with commas
        try:
            return float(val.replace(',', ''))
        except ValueError:
            return None
    return None


def _analyze_source_excel(path: str) -> Dict[str, Any]:
    """Analyze source Excel file and extract statistics."""
    xl = pd.ExcelFile(path)
    result: Dict[str, Any] = {
        "sheets": {},
        "total_rows": 0,
        "total_columns": 0,
        "total_numeric_sum": 0.0,
        "total_numeric_count": 0,
        "all_columns": set(),
    }
    
    for sheet_name in xl.sheet_names:
        # Read without assuming headers to get ALL data
        df_raw = xl.parse(sheet_name, header=None)
        if df_raw.empty:
            continue
        
        # Also read with headers for column analysis
        df = xl.parse(sheet_name)
        
        sheet_info: Dict[str, Any] = {
            "raw_row_count": len(df_raw),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": [str(c) for c in df.columns],
            "numeric_sum": 0.0,
            "numeric_count": 0,
            "numeric_values": [],
        }
        
        # Count all numeric values in raw data
        for col in df_raw.columns:
            for val in df_raw[col]:
                num = _extract_numeric_from_value(val)
                if num is not None:
                    sheet_info["numeric_sum"] += num
                    sheet_info["numeric_count"] += 1
                    sheet_info["numeric_values"].append(num)
        
        result["sheets"][sheet_name] = sheet_info
        result["total_rows"] += sheet_info["row_count"]
        result["total_columns"] += sheet_info["column_count"]
        result["total_numeric_sum"] += sheet_info["numeric_sum"]
        result["total_numeric_count"] += sheet_info["numeric_count"]
        result["all_columns"].update(sheet_info["columns"])
    
    result["sheet_count"] = len(result["sheets"])
    result["all_columns"] = list(result["all_columns"])
    return result


def _analyze_source_csv(path: str) -> Dict[str, Any]:
    """Analyze source CSV file and extract statistics."""
    df = pd.read_csv(path)
    
    result: Dict[str, Any] = {
        "sheets": {
            "data": {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": [str(c) for c in df.columns],
                "numeric_sum": 0.0,
                "numeric_count": 0,
                "numeric_values": [],
            }
        },
        "sheet_count": 1,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_numeric_sum": 0.0,
        "total_numeric_count": 0,
        "all_columns": [str(c) for c in df.columns],
    }
    
    # Count numeric values
    for col in df.columns:
        for val in df[col]:
            num = _extract_numeric_from_value(val)
            if num is not None:
                result["sheets"]["data"]["numeric_sum"] += num
                result["sheets"]["data"]["numeric_count"] += 1
                result["sheets"]["data"]["numeric_values"].append(num)
    
    result["total_numeric_sum"] = result["sheets"]["data"]["numeric_sum"]
    result["total_numeric_count"] = result["sheets"]["data"]["numeric_count"]
    
    return result


def _analyze_extracted_json(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze extracted JSON and extract statistics."""
    result: Dict[str, Any] = {
        "sheets": {},
        "total_rows": 0,
        "total_columns": 0,
        "total_numeric_sum": 0.0,
        "total_numeric_count": 0,
        "all_columns": set(),
    }
    
    # Handle different JSON structures
    
    # Structure 1: {"sheets": {"SheetName": {"data": [...], "columns": [...]}}}
    if "sheets" in json_data:
        sheets_data = json_data["sheets"]
        for sheet_name, sheet_info in sheets_data.items():
            if isinstance(sheet_info, dict):
                data = sheet_info.get("data", [])
                columns = sheet_info.get("columns", [])
                
                sheet_result: Dict[str, Any] = {
                    "row_count": len(data) if isinstance(data, list) else 0,
                    "column_count": len(columns) if isinstance(columns, list) else 0,
                    "columns": columns if isinstance(columns, list) else [],
                    "numeric_sum": 0.0,
                    "numeric_count": 0,
                }
                
                # Count numeric values
                if isinstance(data, list):
                    for row in data:
                        if isinstance(row, dict):
                            for val in row.values():
                                num = _extract_numeric_from_value(val)
                                if num is not None:
                                    sheet_result["numeric_sum"] += num
                                    sheet_result["numeric_count"] += 1
                
                result["sheets"][sheet_name] = sheet_result
                result["total_rows"] += sheet_result["row_count"]
                result["total_columns"] = max(result["total_columns"], sheet_result["column_count"])
                result["total_numeric_sum"] += sheet_result["numeric_sum"]
                result["total_numeric_count"] += sheet_result["numeric_count"]
                result["all_columns"].update(sheet_result["columns"])
    
    # Structure 2: {"extracted_data": {"sheets": ...}}
    elif "extracted_data" in json_data:
        extracted = json_data["extracted_data"]
        if isinstance(extracted, dict) and "sheets" in extracted:
            return _analyze_extracted_json(extracted)
        else:
            # Count all numeric values recursively
            def count_numerics(obj, path=""):
                total_sum = 0.0
                total_count = 0
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        s, c = count_numerics(v, f"{path}.{k}")
                        total_sum += s
                        total_count += c
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        s, c = count_numerics(item, f"{path}[{i}]")
                        total_sum += s
                        total_count += c
                else:
                    num = _extract_numeric_from_value(obj)
                    if num is not None:
                        return num, 1
                return total_sum, total_count
            
            result["total_numeric_sum"], result["total_numeric_count"] = count_numerics(extracted)
    
    # Structure 3: Direct data with period/value pairs
    else:
        def count_all(obj):
            total_sum = 0.0
            total_count = 0
            if isinstance(obj, dict):
                for v in obj.values():
                    s, c = count_all(v)
                    total_sum += s
                    total_count += c
            elif isinstance(obj, list):
                for item in obj:
                    s, c = count_all(item)
                    total_sum += s
                    total_count += c
            else:
                num = _extract_numeric_from_value(obj)
                if num is not None:
                    return num, 1
            return total_sum, total_count
        
        result["total_numeric_sum"], result["total_numeric_count"] = count_all(json_data)
    
    result["sheet_count"] = len(result["sheets"])
    result["all_columns"] = list(result["all_columns"])
    return result


def audit_extraction(
    source_path: str,
    json_data: Optional[Dict[str, Any]] = None,
    json_path: Optional[str] = None,
    tolerance: float = 0.01
) -> AuditResult:
    """
    Audit data extraction by comparing source file with extracted JSON.
    
    Args:
        source_path: Path to source Excel or CSV file
        json_data: Extracted JSON data (dict)
        json_path: Path to JSON file (alternative to json_data)
        tolerance: Tolerance for numeric comparisons
    
    Returns:
        AuditResult with detailed comparison
    """
    # Load JSON if path provided
    if json_data is None and json_path:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    
    if json_data is None:
        raise ValueError("Either json_data or json_path must be provided")
    
    # Detect source file type
    ext = os.path.splitext(source_path)[1].lower()
    
    # Analyze source
    print(f"[AUDIT] Analyzing source file: {source_path}")
    if ext in ['.xls', '.xlsx']:
        source_stats = _analyze_source_excel(source_path)
    elif ext == '.csv':
        source_stats = _analyze_source_csv(source_path)
    else:
        raise ValueError(f"Unsupported source file type: {ext}")
    
    # Analyze JSON
    print(f"[AUDIT] Analyzing extracted JSON...")
    json_stats = _analyze_extracted_json(json_data)
    
    # Create audit result
    result = AuditResult(
        passed=True,
        source_file=source_path,
        json_file=json_path,
        source_row_count=source_stats["total_rows"],
        json_row_count=json_stats["total_rows"],
        source_column_count=source_stats["total_columns"],
        json_column_count=json_stats["total_columns"],
        source_sheet_count=source_stats["sheet_count"],
        json_sheet_count=json_stats["sheet_count"],
        source_numeric_sum=source_stats["total_numeric_sum"],
        json_numeric_sum=json_stats["total_numeric_sum"],
        source_numeric_count=source_stats["total_numeric_count"],
        json_numeric_count=json_stats["total_numeric_count"],
    )
    
    # Check for missing sheets
    source_sheets = set(source_stats["sheets"].keys())
    json_sheets = set(json_stats["sheets"].keys())
    result.missing_sheets = list(source_sheets - json_sheets)
    
    # Check each sheet
    for sheet_name in source_sheets:
        if sheet_name not in json_sheets:
            continue
        
        source_sheet = source_stats["sheets"][sheet_name]
        json_sheet = json_stats["sheets"].get(sheet_name, {})
        
        # Check row counts
        source_rows = source_sheet.get("row_count", 0)
        json_rows = json_sheet.get("row_count", 0)
        if source_rows != json_rows:
            result.row_count_mismatches[sheet_name] = (source_rows, json_rows)
        
        # Check columns
        source_cols = set(source_sheet.get("columns", []))
        json_cols = set(json_sheet.get("columns", []))
        missing = source_cols - json_cols
        if missing:
            result.missing_columns[sheet_name] = list(missing)
    
    # Check numeric totals
    numeric_diff = abs(result.source_numeric_sum - result.json_numeric_sum)
    if numeric_diff > tolerance:
        result.numeric_mismatches.append({
            "location": "total",
            "source_value": result.source_numeric_sum,
            "json_value": result.json_numeric_sum,
            "difference": numeric_diff,
        })
    
    # Check numeric counts
    count_diff = result.source_numeric_count - result.json_numeric_count
    if count_diff > 0:
        result.warnings.append(
            f"Missing {count_diff} numeric values ({result.source_numeric_count} source vs {result.json_numeric_count} json)"
        )
    
    # Determine pass/fail
    result.passed = (
        len(result.missing_sheets) == 0 and
        len(result.row_count_mismatches) == 0 and
        numeric_diff <= tolerance and
        count_diff <= result.source_numeric_count * 0.05  # Allow 5% variance
    )
    
    return result


def audit_and_report(
    source_path: str,
    json_data: Optional[Dict[str, Any]] = None,
    json_path: Optional[str] = None,
    save_report: bool = True
) -> AuditResult:
    """
    Run audit and print/save report.
    
    Args:
        source_path: Path to source Excel or CSV file
        json_data: Extracted JSON data (dict)
        json_path: Path to JSON file (alternative to json_data)
        save_report: Whether to save report to file
    
    Returns:
        AuditResult
    """
    result = audit_extraction(source_path, json_data, json_path)
    result.print_report()
    
    if save_report:
        report_path = os.path.splitext(source_path)[0] + "_audit_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[AUDIT] Report saved to: {report_path}")
    
    return result


# =========================
# CLI ENTRY POINT
# =========================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m app.audit <source_file> <json_file>")
        print("  source_file: Excel (.xlsx, .xls) or CSV (.csv) file")
        print("  json_file: Extracted JSON file to audit")
        sys.exit(1)
    
    source = sys.argv[1]
    json_file = sys.argv[2]
    
    if not os.path.exists(source):
        print(f"Error: Source file not found: {source}")
        sys.exit(1)
    
    if not os.path.exists(json_file):
        print(f"Error: JSON file not found: {json_file}")
        sys.exit(1)
    
    result = audit_and_report(source, json_path=json_file)
    sys.exit(0 if result.passed else 1)
