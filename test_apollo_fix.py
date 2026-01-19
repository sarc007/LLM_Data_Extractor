"""Test Apollo Tyres extraction with fixed prompts."""
import sys
sys.path.insert(0, '.')

from app.page_extractor import process_pdf_page_by_page_v2
import json

print("Testing Apollo Tyres extraction with fixed prompts...")
print("=" * 70)

result = process_pdf_page_by_page_v2(
    'input/Apollo Tyres_Peer 2.pdf',
    iterations_per_page=3,
    combine_iterations=2,
    use_ocr=False
)

print("\n" + "=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)
print(f"Total pages: {result.get('total_pages', 0)}")
print(f"Successful: {result.get('successful_pages', 0)}")
print(f"Doc types: {result.get('document_types_found', [])}")
print(f"Documents produced: {result.get('documents_produced', 0)}")

# Check the combined profit_loss for quality
import os
combined_file = "processed/Apollo Tyres_Peer 2_pages/combined_profit_loss_annual.json"
if os.path.exists(combined_file):
    with open(combined_file) as f:
        combined = json.load(f)
    
    periods = combined.get("data", {}).get("periods", [])
    sales = combined.get("data", {}).get("sales", [])
    
    print(f"\n=== QUALITY CHECK ===")
    print(f"Periods: {periods}")
    print(f"Period count: {len(periods)}")
    print(f"Sales count: {len(sales)}")
    print(f"Sales values: {sales}")
    
    # Check for Trailing/Best/Worst in periods
    invalid = [p for p in periods if any(kw in str(p).lower() for kw in ["trailing", "best", "worst", "ttm"])]
    if invalid:
        print(f"❌ FAIL: Found invalid periods: {invalid}")
    else:
        print(f"✅ PASS: No Trailing/Best/Worst in periods")
    
    # Check array length consistency
    data = combined.get("data", {})
    mismatches = []
    for key, value in data.items():
        if isinstance(value, list) and key != "periods":
            if len(value) != len(periods):
                mismatches.append(f"{key}: {len(value)} vs {len(periods)}")
    
    if mismatches:
        print(f"❌ FAIL: Array length mismatches: {mismatches}")
    else:
        print(f"✅ PASS: All arrays have {len(periods)} values")
