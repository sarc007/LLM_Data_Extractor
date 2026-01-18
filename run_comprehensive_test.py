"""
Comprehensive automated test for all PDFs in input folder.
Generates detailed report on extraction quality and issues.
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.page_extractor import process_pdf_page_by_page_v2, get_pdf_page_count

def analyze_extraction_quality(result: Dict) -> Dict[str, Any]:
    """Analyze the quality of extracted data."""
    issues = []
    warnings = []
    stats = {
        "total_documents": 0,
        "documents_with_periods": 0,
        "documents_without_periods": 0,
        "null_values_count": 0,
        "empty_arrays_count": 0,
        "period_types": set(),
        "document_types": set()
    }
    
    documents = result.get("documents", [])
    stats["total_documents"] = len(documents)
    
    for doc in documents:
        doc_type = doc.get("document_type", "unknown")
        period_type = doc.get("period_type", "unknown")
        periods = doc.get("periods", [])
        data = doc.get("data", {})
        
        stats["document_types"].add(doc_type)
        stats["period_types"].add(period_type)
        
        if periods:
            stats["documents_with_periods"] += 1
        else:
            stats["documents_without_periods"] += 1
            if doc_type not in ["metadata", "shareholding_pattern"]:
                issues.append(f"{doc_type}: Missing periods")
        
        # Check for null values and empty arrays
        def count_nulls(obj, path=""):
            count = 0
            if obj is None:
                return 1
            if isinstance(obj, dict):
                for k, v in obj.items():
                    count += count_nulls(v, f"{path}.{k}")
            elif isinstance(obj, list):
                if len(obj) == 0:
                    stats["empty_arrays_count"] += 1
                for i, v in enumerate(obj):
                    count += count_nulls(v, f"{path}[{i}]")
            return count
        
        stats["null_values_count"] += count_nulls(data)
        
        # Check period consistency
        if periods and isinstance(data, dict):
            expected_len = len(periods)
            for key, value in data.items():
                if isinstance(value, list) and key != "periods":
                    if len(value) != expected_len:
                        warnings.append(f"{doc_type}.{key}: Expected {expected_len} values, got {len(value)}")
    
    # Convert sets to lists for JSON serialization
    stats["document_types"] = list(stats["document_types"])
    stats["period_types"] = list(stats["period_types"])
    
    return {
        "stats": stats,
        "issues": issues,
        "warnings": warnings,
        "quality_score": calculate_quality_score(stats, issues, warnings)
    }


def calculate_quality_score(stats: Dict, issues: List, warnings: List) -> float:
    """Calculate a quality score from 0-100."""
    score = 100.0
    
    # Deduct for issues
    score -= len(issues) * 10
    score -= len(warnings) * 2
    
    # Deduct for null values
    score -= min(stats["null_values_count"] * 0.5, 20)
    
    # Deduct for empty arrays
    score -= min(stats["empty_arrays_count"] * 1, 10)
    
    # Bonus for having multiple document types
    if len(stats["document_types"]) >= 3:
        score += 5
    
    return max(0, min(100, score))


def test_single_pdf(pdf_path: str) -> Dict[str, Any]:
    """Test extraction on a single PDF."""
    pdf_name = Path(pdf_path).stem
    print(f"\n{'='*70}")
    print(f"TESTING: {pdf_name}")
    print(f"{'='*70}")
    
    start_time = datetime.now()
    
    try:
        # Get page count
        page_count = get_pdf_page_count(pdf_path)
        print(f"Pages: {page_count}")
        
        # Run extraction
        result = process_pdf_page_by_page_v2(
            pdf_path=pdf_path,
            iterations_per_page=3,
            combine_iterations=2,
            use_ocr=False  # Faster for testing
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Analyze quality
        quality = analyze_extraction_quality(result)
        
        return {
            "pdf_name": pdf_name,
            "status": "success",
            "page_count": page_count,
            "duration_seconds": duration,
            "documents_extracted": len(result.get("documents", [])),
            "document_types": quality["stats"]["document_types"],
            "period_types": quality["stats"]["period_types"],
            "quality_score": quality["quality_score"],
            "issues": quality["issues"],
            "warnings": quality["warnings"][:5],  # Limit warnings
            "null_values": quality["stats"]["null_values_count"]
        }
        
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return {
            "pdf_name": pdf_name,
            "status": "error",
            "error": str(e),
            "duration_seconds": duration,
            "quality_score": 0
        }


def run_all_tests():
    """Run tests on all PDFs in input folder."""
    input_dir = Path("input")
    pdf_files = list(input_dir.glob("*.pdf"))
    
    print("\n" + "="*70)
    print("COMPREHENSIVE PDF EXTRACTION TEST")
    print("="*70)
    print(f"Found {len(pdf_files)} PDF files to test")
    print(f"Started: {datetime.now().isoformat()}")
    
    results = []
    
    for pdf_path in pdf_files:
        result = test_single_pdf(str(pdf_path))
        results.append(result)
    
    # Generate summary
    summary = {
        "test_date": datetime.now().isoformat(),
        "total_pdfs": len(results),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "error"]),
        "average_quality_score": sum(r["quality_score"] for r in results) / len(results) if results else 0,
        "total_documents_extracted": sum(r.get("documents_extracted", 0) for r in results),
        "results": results
    }
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total PDFs: {summary['total_pdfs']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Average Quality Score: {summary['average_quality_score']:.1f}/100")
    print(f"Total Documents Extracted: {summary['total_documents_extracted']}")
    
    print("\n" + "-"*70)
    print("INDIVIDUAL RESULTS:")
    print("-"*70)
    
    for r in results:
        status_icon = "✅" if r["status"] == "success" else "❌"
        print(f"\n{status_icon} {r['pdf_name']}")
        if r["status"] == "success":
            print(f"   Quality: {r['quality_score']:.0f}/100 | Docs: {r['documents_extracted']} | Time: {r['duration_seconds']:.1f}s")
            print(f"   Types: {r['document_types']}")
            if r["issues"]:
                print(f"   Issues: {r['issues']}")
        else:
            print(f"   Error: {r['error']}")
    
    # Save report
    report_path = Path("processed") / "comprehensive_test_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[SAVED] Report: {report_path}")
    
    return summary


if __name__ == "__main__":
    run_all_tests()
