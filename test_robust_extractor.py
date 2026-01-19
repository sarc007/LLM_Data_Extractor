"""Test the robust multi-extractor pipeline."""
import sys
sys.path.insert(0, '.')

from app.robust_extractor import RobustExtractor, process_page_robust
import json

print("=" * 70)
print("TESTING ROBUST MULTI-EXTRACTOR PIPELINE")
print("=" * 70)

# Test 1: Multi-source extraction
print("\n[TEST 1] Multi-source extraction on Apollo Tyres page 1...")
extractor = RobustExtractor()
result = extractor.extract_all_sources('input/Apollo Tyres_Peer 2.pdf', 1)

print(f"\nExtractors used: {result['extractors_used']}")
for name, data in result['individual_results'].items():
    print(f"  {name}: {data.get('char_count', 0)} chars, error: {data.get('error', 'none')}")

combined = result['combined']
print(f"\nCombined result:")
print(f"  Best source: {combined['best_source']}")
print(f"  Total chars: {combined['char_count']}")
print(f"  Unique numbers: {combined['number_count']}")
print(f"  High-confidence numbers: {len(combined['high_confidence_numbers'])}")

# Show sample numbers
print(f"\nSample high-confidence numbers:")
for num in sorted(combined['high_confidence_numbers'], reverse=True)[:15]:
    print(f"  {num:,.2f}")

# Test 2: Full robust extraction with 7 iterations
print("\n" + "=" * 70)
print("[TEST 2] Full robust extraction with validation (7 iterations max)...")
print("=" * 70)

robust_result = process_page_robust(
    'input/Apollo Tyres_Peer 2.pdf',
    page_num=1,
    doc_type="profit_loss",
    target_periods=["Mar-16", "Mar-17", "Mar-18", "Mar-19", "Mar-20", 
                    "Mar-21", "Mar-22", "Mar-23", "Mar-24", "Mar-25"],
    max_iterations=7
)

print("\n" + "=" * 70)
print("EXTRACTION RESULT")
print("=" * 70)

if "error" in robust_result:
    print(f"❌ Error: {robust_result['error']}")
else:
    print(f"\nValidation:")
    val = robust_result['validation']
    print(f"  Accuracy: {val['accuracy_percent']}%")
    print(f"  Extracted significant: {val['extracted_significant']}")
    print(f"  Matched in source: {val['matched_in_source']}")
    print(f"  Fields with data: {val['fields_with_data']}")
    print(f"  Null percent: {val['null_percent']}%")
    print(f"  Array issues: {val['array_issues']}")
    print(f"  PASSED: {val['passed']}")
    
    print(f"\nIterations: {robust_result['total_iterations']}")
    for it in robust_result['iterations']:
        status = "✅" if it['passed'] else "⏳"
        print(f"  {status} Iteration {it['iteration']}: {it['accuracy']:.1f}% accuracy, {it['fields']} fields, {it['null_pct']:.1f}% nulls")
    
    print(f"\nExtracted data preview:")
    data = robust_result['extracted_data']
    print(f"  Periods: {data.get('periods', [])}")
    print(f"  Sales: {data.get('sales', [])}")
    print(f"  Net Profit: {data.get('net_profit', [])}")
    print(f"  EPS: {data.get('eps', [])}")

# Test 3: Scenarios extraction
print("\n" + "=" * 70)
print("[TEST 3] Scenarios (Best/Worst Case) extraction...")
print("=" * 70)

scenarios_result = process_page_robust(
    'input/Apollo Tyres_Peer 2.pdf',
    page_num=1,
    doc_type="scenarios",
    target_periods=["Best Case", "Worst Case"],
    max_iterations=5
)

if "error" not in scenarios_result:
    print(f"\nScenarios extraction:")
    data = scenarios_result['extracted_data']
    print(f"  Periods: {data.get('periods', [])}")
    print(f"  Sales: {data.get('sales', [])}")
    print(f"  Net Profit: {data.get('net_profit', [])}")
    print(f"  EPS: {data.get('eps', [])}")
    print(f"\n  Validation accuracy: {scenarios_result['validation']['accuracy_percent']}%")
