#!/usr/bin/env python
"""Test page-by-page extraction on all PDFs in input folder."""
import json
from pathlib import Path
from app.page_extractor import process_pdf_page_by_page_v2

def test_all_pdfs():
    input_dir = Path('input')
    pdfs = list(input_dir.glob('*.pdf'))
    
    print(f"Found {len(pdfs)} PDF files to test")
    print("=" * 70)
    
    results = []
    
    for pdf_path in pdfs:
        print(f"\n{'='*70}")
        print(f"TESTING: {pdf_path.name}")
        print(f"{'='*70}")
        
        try:
            result = process_pdf_page_by_page_v2(
                pdf_path=str(pdf_path),
                iterations_per_page=3,
                combine_iterations=3,
                use_ocr=True
            )
            
            results.append({
                'file': pdf_path.name,
                'status': 'SUCCESS',
                'total_pages': result['total_pages'],
                'pages_successful': result['pages_successful'],
                'document_types': result['document_types_found'],
                'documents_produced': len(result['documents'])
            })
            
        except Exception as e:
            results.append({
                'file': pdf_path.name,
                'status': 'ERROR',
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 70)
    print("BATCH TEST SUMMARY")
    print("=" * 70)
    
    success = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"SUCCESS: {success}/{len(results)}")
    
    for r in results:
        if r['status'] == 'SUCCESS':
            print(f"  ✅ {r['file']}: {r['total_pages']} pages, {r['documents_produced']} documents")
            print(f"     Types: {r['document_types']}")
        else:
            print(f"  ❌ {r['file']}: {r['error']}")
    
    return results

if __name__ == '__main__':
    test_all_pdfs()
