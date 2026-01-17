#!/usr/bin/env python3
"""
Standalone script to run Qwen3 480B extraction pipeline.

Usage:
    python run_qwen_extraction.py <file_path> [options]

Options:
    --convert-pdf       Convert Excel/CSV to PDF before processing
    --iterations N      Number of self-check iterations (default: 5)
    --output PATH       Custom output path for results JSON
    --paddleocr         Use PaddleOCR for PDF extraction (better for scanned docs)
    --ocr-dpi N         DPI for OCR conversion (default: 200)
    --no-audit          Skip audit verification for Excel/CSV

Examples:
    python run_qwen_extraction.py "input/JK tyres sales.pdf"
    python run_qwen_extraction.py "input/JK tyres sales.pdf" --paddleocr
    python run_qwen_extraction.py "data/sales.xlsx" --convert-pdf
    python run_qwen_extraction.py "data/report.csv" --iterations 3
    python run_qwen_extraction.py "data/report.xlsx"  # auto audit enabled
"""

import sys
import os
import json
import argparse
from datetime import datetime

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.qwen_pipeline import (
    process_document_qwen,
    QWEN_MODEL,
    OLLAMA_HOST,
    PADDLEOCR_AVAILABLE,
    AUDIT_AVAILABLE
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract data from documents using Qwen3 480B Cloud"
    )
    parser.add_argument(
        "file_path",
        help="Path to the document (PDF, Excel, or CSV)"
    )
    parser.add_argument(
        "--convert-pdf",
        action="store_true",
        help="Convert Excel/CSV to PDF before processing"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of self-check iterations (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for results JSON"
    )
    parser.add_argument(
        "--paddleocr",
        action="store_true",
        help="Use PaddleOCR for PDF extraction (better for scanned docs)"
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=200,
        help="DPI for OCR conversion (default: 200)"
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Skip audit verification for Excel/CSV"
    )
    
    args = parser.parse_args()
    
    # Validate file exists
    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("QWEN3 480B DATA EXTRACTION")
    print("=" * 60)
    print(f"Model: {QWEN_MODEL}")
    print(f"Host: {OLLAMA_HOST}")
    print(f"Input: {args.file_path}")
    print(f"Iterations: {args.iterations}")
    print(f"Convert to PDF: {args.convert_pdf}")
    print(f"PaddleOCR: {args.paddleocr} (available: {PADDLEOCR_AVAILABLE})")
    print(f"Audit: {not args.no_audit} (available: {AUDIT_AVAILABLE})")
    if args.paddleocr:
        print(f"OCR DPI: {args.ocr_dpi}")
    
    # Run extraction
    try:
        extracted_data, analysis = process_document_qwen(
            args.file_path,
            convert_to_pdf_first=args.convert_pdf,
            iterations=args.iterations,
            use_paddleocr=args.paddleocr,
            run_audit=not args.no_audit,
            ocr_dpi=args.ocr_dpi
        )
    except Exception as e:
        print(f"\nError during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Prepare output
    result = {
        "extracted_data": extracted_data,
        "analysis": analysis,
        "metadata": {
            "source_file": os.path.basename(args.file_path),
            "model": QWEN_MODEL,
            "iterations": args.iterations,
            "processed_at": datetime.now().isoformat()
        }
    }
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Save to processed/ directory
        os.makedirs("processed", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.file_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"processed/{base_name}_qwen_{timestamp}.json"
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'=' * 60}")
    print("RESULTS SAVED")
    print("=" * 60)
    print(f"Output: {output_path}")
    
    # Print analysis summary - the 4 key points
    print("\n" + "-" * 40)
    print("ANALYSIS - 4 KEY POINTS")
    print("-" * 40)
    
    # 1. Data Type
    if "data_type" in analysis:
        dt = analysis["data_type"]
        if isinstance(dt, dict):
            print(f"\n1. DATA TYPE: {dt.get('primary_type', 'unknown')}")
            if dt.get('confidence'):
                conf = dt['confidence']
                if isinstance(conf, (int, float)):
                    print(f"   Confidence: {conf:.0%}")
            if dt.get('reasoning'):
                print(f"   Reasoning: {dt['reasoning']}")
        else:
            print(f"\n1. DATA TYPE: {dt}")
    
    # 2. Number of Periods
    print(f"\n2. NUMBER OF PERIODS: {analysis.get('number_of_periods', 'unknown')}")
    
    # 3. Period Type
    print(f"\n3. PERIOD TYPE: {analysis.get('period_type', 'unknown')}")
    if analysis.get('periods_found'):
        periods = analysis['periods_found']
        if isinstance(periods, list):
            print(f"   Periods: {', '.join(str(p) for p in periods[:10])}")
            if len(periods) > 10:
                print(f"   ... and {len(periods) - 10} more")
    
    # 4. Sample Queries
    if "sample_queries" in analysis and analysis["sample_queries"]:
        print(f"\n4. SAMPLE QUERIES:")
        for i, query in enumerate(analysis["sample_queries"][:8], 1):
            print(f"   {i}. {query}")
    
    # 5. Audit Result (if available)
    if "audit" in analysis:
        audit = analysis["audit"]
        status = "✅ PASSED" if audit.get("passed") else "❌ FAILED"
        print(f"\n5. AUDIT VERIFICATION: {status}")
        print(f"   Source rows: {audit.get('source_rows', 'N/A')}")
        print(f"   JSON rows: {audit.get('json_rows', 'N/A')}")
        print(f"   Source numeric sum: {audit.get('source_numeric_sum', 'N/A'):,.2f}")
        print(f"   JSON numeric sum: {audit.get('json_numeric_sum', 'N/A'):,.2f}")
        if audit.get("warnings"):
            print("   Warnings:")
            for warning in audit["warnings"]:
                print(f"     ⚠️  {warning}")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60 + "\n")
    
    return result


if __name__ == "__main__":
    main()
