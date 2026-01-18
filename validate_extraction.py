#!/usr/bin/env python
"""
Validate extraction output against source PDF.

Usage:
    python validate_extraction.py <pdf_path> <json_path>
    python validate_extraction.py input/Apollo\ Tyres_Peer\ 2.pdf processed/output.json
"""
import sys
import json
import re
import pdfplumber
from pathlib import Path


def get_pdf_periods(pdf_path: str) -> list:
    """Extract actual periods from PDF column headers."""
    periods = []
    period_pattern = re.compile(r'(Mar|Jun|Sep|Dec)-(\d{2})')
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:3]:
            text = page.extract_text() or ""
            for month, year in period_pattern.findall(text):
                period = f"{month}-{year}"
                if period not in periods:
                    periods.append(period)
    return periods


def validate(pdf_path: str, json_path: str) -> list:
    """Validate JSON extraction against source PDF."""
    issues = []
    
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    extracted = data.get("extracted_data", data)
    pl = extracted.get("profit_loss", {})
    
    # Get PDF periods
    pdf_periods = get_pdf_periods(pdf_path)
    print(f"\n[PDF] Periods found: {pdf_periods[:10]}...")
    
    # Get JSON periods
    sales = pl.get("sales", [])
    json_periods = [s.get("period") for s in sales if s.get("period")]
    print(f"[JSON] Periods in sales: {json_periods[:10]}...")
    
    # 1. Check for invented periods
    invented = [p for p in json_periods if p not in pdf_periods]
    if invented:
        issues.append(f"❌ INVENTED PERIODS: {invented}")
    else:
        print("✅ No invented periods")
    
    # 2. Check first period matches
    if json_periods and pdf_periods:
        if json_periods[0] != pdf_periods[0]:
            issues.append(f"❌ FIRST PERIOD MISMATCH: JSON={json_periods[0]}, PDF={pdf_periods[0]}")
        else:
            print(f"✅ First period matches: {json_periods[0]}")
    
    # 3. Check for frequency mixing (70%+ drops)
    for i in range(1, len(sales)):
        prev = sales[i-1].get("value", 0) or 0
        curr = sales[i].get("value", 0) or 0
        if prev > 0 and curr > 0:
            drop = (prev - curr) / prev * 100
            if drop > 70:
                issues.append(
                    f"❌ FREQUENCY MIX: {sales[i-1]['period']}={prev} → {sales[i]['period']}={curr} ({drop:.1f}% drop)"
                )
    if not any("FREQUENCY" in i for i in issues):
        print("✅ No frequency mixing detected")
    
    # 4. Check structural consistency
    counts = {k: len(v) for k, v in pl.items() if isinstance(v, list)}
    if counts:
        max_c, min_c = max(counts.values()), min(counts.values())
        if max_c - min_c > 5:
            issues.append(f"❌ INCONSISTENT COUNTS: {counts}")
        else:
            print(f"✅ Consistent period counts: {min_c}-{max_c}")
    
    # 5. Check completeness
    for metric in ["operating_profit", "expenses", "net_profit"]:
        values = pl.get(metric, [])
        if len(values) < 5:
            issues.append(f"❌ INCOMPLETE: {metric} has only {len(values)} values")
        else:
            print(f"✅ {metric}: {len(values)} values")
    
    return issues


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    json_path = sys.argv[2]
    
    if not Path(pdf_path).exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)
    if not Path(json_path).exists():
        print(f"JSON not found: {json_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"VALIDATING EXTRACTION")
    print(f"{'='*60}")
    print(f"PDF:  {pdf_path}")
    print(f"JSON: {json_path}")
    print(f"{'='*60}")
    
    issues = validate(pdf_path, json_path)
    
    print(f"\n{'='*60}")
    if issues:
        print(f"ISSUES FOUND: {len(issues)}")
        print(f"{'='*60}")
        for issue in issues:
            print(issue)
        sys.exit(1)
    else:
        print("ALL VALIDATIONS PASSED ✅")
        print(f"{'='*60}")
        sys.exit(0)


if __name__ == "__main__":
    main()
