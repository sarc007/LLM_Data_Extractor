#!/usr/bin/env python
"""Test extraction on all PDF files in input folder."""
import os
import json
from pathlib import Path
from app.qwen_pipeline import process_document_qwen
from validate_extraction import get_pdf_periods, get_json_periods, get_metric_values

def test_all_pdfs():
    input_dir = Path('input')
    output_dir = Path('processed/batch_test')
    output_dir.mkdir(exist_ok=True)

    pdfs = list(input_dir.glob('*.pdf'))
    print(f'Found {len(pdfs)} PDF files to test')
    print('='*60)

    results = []
    for pdf_path in pdfs:
        pdf_name = pdf_path.name
        print(f'\n[{pdf_name}]')
        
        try:
            # Get PDF periods first
            pdf_periods = get_pdf_periods(str(pdf_path))
            print(f'  PDF periods: {pdf_periods[:5]}...')
            
            # Run extraction
            print(f'  Extracting...')
            data, analysis = process_document_qwen(str(pdf_path), iterations=2, run_audit=False)
            
            # Save output
            output_file = output_dir / f'{pdf_path.stem}_output.json'
            with open(output_file, 'w') as f:
                json.dump({'extracted_data': data, 'analysis': analysis}, f, indent=2)
            
            # Quick validation
            pl = data.get('profit_loss', {})
            periods = pl.get('periods', [])
            sales = pl.get('sales', [])
            
            # Check for issues
            issues = []
            if not periods:
                issues.append('No periods extracted')
            elif periods and pdf_periods:
                invented = [p for p in periods if p not in pdf_periods]
                if invented:
                    issues.append(f'Invented periods: {invented}')
            
            if len(sales) < 5:
                issues.append(f'Sales incomplete: {len(sales)} values')
            
            status = 'PASS' if not issues else 'ISSUES'
            results.append({
                'file': pdf_name,
                'status': status,
                'periods': len(periods),
                'sales': len(sales),
                'issues': issues
            })
            
            print(f'  Status: {status}')
            print(f'  Periods: {len(periods)}, Sales: {len(sales)} values')
            if issues:
                for issue in issues:
                    print(f'  - {issue}')
                    
        except Exception as e:
            results.append({'file': pdf_name, 'status': 'ERROR', 'error': str(e)})
            print(f'  ERROR: {e}')

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    passed = sum(1 for r in results if r['status'] == 'PASS')
    print(f'PASSED: {passed}/{len(results)}')
    for r in results:
        status_icon = 'PASS' if r['status'] == 'PASS' else 'FAIL'
        print(f"[{status_icon}] {r['file']}: {r['status']}")
        if r.get('issues'):
            for issue in r['issues']:
                print(f'   - {issue}')
        if r.get('error'):
            print(f'   - {r["error"]}')
    
    return results

if __name__ == '__main__':
    test_all_pdfs()
