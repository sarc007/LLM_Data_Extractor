"""Test scenarios extraction for Best/Worst Case data."""
import sys
sys.path.insert(0, '.')

from app.page_extractor import detect_document_types, extract_page_data
import pdfplumber
import json

print("Testing scenarios extraction for Best/Worst Case...")
print("=" * 70)

# Extract page 1 text
with pdfplumber.open('input/Apollo Tyres_Peer 2.pdf') as pdf:
    page = pdf.pages[0]
    text = page.extract_text() or ""

print(f"Page 1 text length: {len(text)} chars")
print("\nDetecting document types...")

# Detect document types
doc_types = detect_document_types(text)
print(f"\nFound {len(doc_types)} document types:")
for doc in doc_types:
    print(f"  - {doc['document_type']} ({doc.get('period_type', 'none')}): {doc.get('periods', [])[:5]}...")

# Check if scenarios was detected
scenarios_doc = next((d for d in doc_types if d['document_type'] == 'scenarios'), None)
if scenarios_doc:
    print(f"\n✅ SCENARIOS document type detected!")
    print(f"   Periods: {scenarios_doc.get('periods', [])}")
    
    # Extract scenarios data - test raw LLM output
    print("\nExtracting scenarios data (bypassing validation)...")
    from app.page_extractor import _build_page_extraction_prompt, _call_llm, _extract_json_from_text
    
    prompt = _build_page_extraction_prompt(text, "scenarios", scenarios_doc.get('periods'))
    print(f"\nPrompt snippet: ...{prompt[-500:]}")
    
    response = _call_llm(prompt)
    raw_json = _extract_json_from_text(response)
    print(f"\nRaw LLM extraction (before validation):")
    print(json.dumps(raw_json, indent=2)[:2000] if raw_json else "None")
else:
    print(f"\n❌ SCENARIOS document type NOT detected")
    print("   The LLM may need prompting to recognize Best/Worst Case columns")
