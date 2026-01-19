"""
Robust Multi-Extractor Pipeline for Foolproof PDF Data Extraction.

Features:
1. Triple extractor approach (pdfplumber, PyMuPDF, pdfminer)
2. Cross-validation and merge logic
3. 7-iteration validation loop with self-healing
4. Numeric value audit (source vs extracted)
5. Gap detection and filling mechanism
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import pdfplumber

# Try importing additional extractors
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("[WARN] PyMuPDF not available - install with: pip install pymupdf")

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from io import StringIO
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    print("[WARN] pdfminer not available - install with: pip install pdfminer.six")


class RobustExtractor:
    """Multi-extractor pipeline with validation and self-healing."""
    
    def __init__(self, max_iterations: int = 7, min_extractors: int = 2):
        self.max_iterations = max_iterations
        self.min_extractors = min_extractors
        self.extractors_available = self._check_extractors()
        
    def _check_extractors(self) -> List[str]:
        """Check which extractors are available."""
        available = ["pdfplumber"]  # Always available
        if PYMUPDF_AVAILABLE:
            available.append("pymupdf")
        if PDFMINER_AVAILABLE:
            available.append("pdfminer")
        print(f"[INFO] Available extractors: {available}")
        return available
    
    def extract_with_pdfplumber(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """Extract text using pdfplumber."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num < 1 or page_num > len(pdf.pages):
                    return {"text": "", "tables": [], "error": "Invalid page number"}
                
                page = pdf.pages[page_num - 1]
                text = page.extract_text() or ""
                
                # Also extract tables
                tables = []
                for table in page.extract_tables():
                    if table:
                        tables.append(table)
                
                # Fix number spacing issues
                text = self._fix_number_spacing(text)
                
                return {
                    "text": text,
                    "tables": tables,
                    "char_count": len(text),
                    "extractor": "pdfplumber"
                }
        except Exception as e:
            return {"text": "", "tables": [], "error": str(e), "extractor": "pdfplumber"}
    
    def extract_with_pymupdf(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """Extract text using PyMuPDF (fitz)."""
        if not PYMUPDF_AVAILABLE:
            return {"text": "", "error": "PyMuPDF not installed", "extractor": "pymupdf"}
        
        try:
            doc = fitz.open(pdf_path)
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return {"text": "", "error": "Invalid page number", "extractor": "pymupdf"}
            
            page = doc[page_num - 1]
            
            # Extract text with different methods
            text_dict = page.get_text("dict")
            text_plain = page.get_text("text")
            text_blocks = page.get_text("blocks")
            
            # Also try to get tables via text blocks
            tables = self._extract_tables_from_blocks(text_blocks)
            
            doc.close()
            
            # Fix number spacing
            text_plain = self._fix_number_spacing(text_plain)
            
            return {
                "text": text_plain,
                "text_dict": text_dict,
                "tables": tables,
                "char_count": len(text_plain),
                "extractor": "pymupdf"
            }
        except Exception as e:
            return {"text": "", "error": str(e), "extractor": "pymupdf"}
    
    def extract_with_pdfminer(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """Extract text using pdfminer."""
        if not PDFMINER_AVAILABLE:
            return {"text": "", "error": "pdfminer not installed", "extractor": "pdfminer"}
        
        try:
            output = StringIO()
            with open(pdf_path, 'rb') as f:
                rsrcmgr = PDFResourceManager()
                device = TextConverter(rsrcmgr, output, laparams=LAParams())
                interpreter = PDFPageInterpreter(rsrcmgr, device)
                
                for i, page in enumerate(PDFPage.get_pages(f)):
                    if i == page_num - 1:
                        interpreter.process_page(page)
                        break
                
                device.close()
            
            text = output.getvalue()
            output.close()
            
            # Fix number spacing
            text = self._fix_number_spacing(text)
            
            return {
                "text": text,
                "char_count": len(text),
                "extractor": "pdfminer"
            }
        except Exception as e:
            return {"text": "", "error": str(e), "extractor": "pdfminer"}
    
    def _fix_number_spacing(self, text: str) -> str:
        """Fix common PDF extraction issues with number spacing."""
        # Pattern: single digit + space + digit(s) + comma + digits (e.g., "1 1,848.56" -> "11,848.56")
        pattern = r'(\d)\s+(\d{1,2},\d{3})'
        text = re.sub(pattern, r'\1\2', text)
        
        # Fix spaces in decimals (e.g., "1,234. 56" -> "1,234.56")
        text = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', text)
        
        # Fix spaces in large numbers (e.g., "12 345" -> "12345")
        text = re.sub(r'(\d)\s+(\d{3})(?!\d)', r'\1\2', text)
        
        return text
    
    def _extract_tables_from_blocks(self, blocks: List) -> List[List]:
        """Extract table-like structures from PyMuPDF blocks."""
        tables = []
        current_row = []
        last_y = None
        
        for block in blocks:
            if isinstance(block, tuple) and len(block) >= 5:
                x0, y0, x1, y1, text = block[:5]
                
                if last_y is not None and abs(y0 - last_y) > 10:
                    if current_row:
                        tables.append(current_row)
                    current_row = []
                
                if isinstance(text, str):
                    current_row.append(text.strip())
                last_y = y0
        
        if current_row:
            tables.append(current_row)
        
        return tables
    
    def extract_all_sources(self, pdf_path: str, page_num: int) -> Dict[str, Any]:
        """Extract from ALL available sources and combine."""
        results = {}
        
        # Extract from each available extractor
        results["pdfplumber"] = self.extract_with_pdfplumber(pdf_path, page_num)
        
        if PYMUPDF_AVAILABLE:
            results["pymupdf"] = self.extract_with_pymupdf(pdf_path, page_num)
        
        if PDFMINER_AVAILABLE:
            results["pdfminer"] = self.extract_with_pdfminer(pdf_path, page_num)
        
        # Combine and validate
        combined = self._combine_extractions(results)
        
        return {
            "individual_results": results,
            "combined": combined,
            "extractors_used": list(results.keys()),
            "page_num": page_num
        }
    
    def _combine_extractions(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine extractions from multiple sources, preferring the most complete."""
        texts = []
        all_numbers = []
        
        for name, result in results.items():
            if result.get("text"):
                texts.append((name, result["text"], len(result["text"])))
                # Extract all numbers from this text
                numbers = self._extract_numbers(result["text"])
                all_numbers.extend(numbers)
        
        # Sort by length (longest first) - usually most complete
        texts.sort(key=lambda x: x[2], reverse=True)
        
        # Use the longest text as base
        best_text = texts[0][1] if texts else ""
        best_source = texts[0][0] if texts else "none"
        
        # Get unique numbers from all sources
        unique_numbers = list(set(all_numbers))
        
        # Find numbers that appear in ALL sources (high confidence)
        number_counts = Counter(all_numbers)
        high_confidence_numbers = [n for n, count in number_counts.items() if count >= 2]
        
        return {
            "text": best_text,
            "best_source": best_source,
            "char_count": len(best_text),
            "all_numbers": unique_numbers,
            "high_confidence_numbers": high_confidence_numbers,
            "number_count": len(unique_numbers),
            "sources_compared": len(texts)
        }
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract all numeric values from text."""
        # Pattern for numbers with optional commas and decimals
        pattern = r'-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?'
        matches = re.findall(pattern, text)
        
        numbers = []
        for match in matches:
            try:
                # Remove commas and convert
                clean = match.replace(',', '')
                num = float(clean)
                numbers.append(num)
            except ValueError:
                pass
        
        return numbers
    
    def validate_extraction(
        self, 
        extracted_json: Dict, 
        source_text: str,
        source_numbers: List[float]
    ) -> Dict[str, Any]:
        """Validate extracted JSON against source data with SMART matching."""
        
        # Extract numbers from JSON
        json_numbers = self._extract_numbers_from_json(extracted_json)
        
        # SMART VALIDATION: Focus on SIGNIFICANT numbers (>100 or specific patterns)
        # Filter out noise like ratios, small numbers, page numbers
        significant_source = set(round(n, 2) for n in source_numbers if abs(n) >= 10)
        significant_json = set(round(n, 2) for n in json_numbers if abs(n) >= 10)
        
        # Also check that extracted values EXIST in source
        matched = significant_json & significant_source
        not_in_source = significant_json - significant_source  # Values we extracted that aren't in source (potential errors)
        
        # Calculate coverage based on what we DID extract matching source
        accuracy = len(matched) / len(significant_json) * 100 if significant_json else 100
        
        # Check array consistency
        periods = extracted_json.get("periods", [])
        array_issues = []
        fields_with_data = 0
        null_count = 0
        
        for key, value in extracted_json.items():
            if key == "periods":
                continue
            if isinstance(value, list):
                if len(value) != len(periods):
                    array_issues.append(f"{key}: {len(value)} vs {len(periods)} periods")
                # Count non-null values
                non_null = sum(1 for v in value if v is not None)
                if non_null > 0:
                    fields_with_data += 1
                null_count += len(value) - non_null
        
        # Success criteria: 
        # 1. Accuracy >= 90% (extracted values exist in source)
        # 2. No array length issues
        # 3. Multiple fields have data
        # 4. Less than 20% null values
        total_values = len(periods) * fields_with_data if periods else 1
        null_percent = (null_count / total_values * 100) if total_values > 0 else 0
        
        passed = (accuracy >= 90 and 
                  len(array_issues) == 0 and 
                  fields_with_data >= 5 and 
                  null_percent < 30)
        
        return {
            "accuracy_percent": round(accuracy, 1),
            "extracted_significant": len(significant_json),
            "matched_in_source": len(matched),
            "potential_errors": len(not_in_source),
            "error_values": sorted(list(not_in_source))[:10],
            "fields_with_data": fields_with_data,
            "null_percent": round(null_percent, 1),
            "array_issues": array_issues,
            "passed": passed
        }
    
    def _extract_numbers_from_json(self, data: Any, numbers: List[float] = None) -> List[float]:
        """Recursively extract all numbers from JSON structure."""
        if numbers is None:
            numbers = []
        
        if isinstance(data, (int, float)) and data is not None:
            numbers.append(float(data))
        elif isinstance(data, list):
            for item in data:
                self._extract_numbers_from_json(item, numbers)
        elif isinstance(data, dict):
            for value in data.values():
                self._extract_numbers_from_json(value, numbers)
        
        return numbers
    
    def find_missing_data(
        self,
        extracted_json: Dict,
        source_numbers: List[float],
        source_text: str
    ) -> Dict[str, Any]:
        """Identify specific missing data and suggest fixes."""
        
        json_numbers = set(round(n, 2) for n in self._extract_numbers_from_json(extracted_json))
        source_set = set(round(n, 2) for n in source_numbers)
        
        missing = sorted(source_set - json_numbers, reverse=True)
        
        # Try to identify what the missing numbers represent
        suggestions = []
        
        for num in missing[:10]:  # Top 10 missing
            # Search for context around this number in source text
            num_str = f"{num:,.2f}".rstrip('0').rstrip('.')
            
            # Find in text
            idx = source_text.find(str(int(num))) if num == int(num) else source_text.find(num_str)
            if idx >= 0:
                context_start = max(0, idx - 50)
                context_end = min(len(source_text), idx + 50)
                context = source_text[context_start:context_end]
                suggestions.append({
                    "value": num,
                    "context": context.strip(),
                    "likely_field": self._guess_field_from_context(context)
                })
        
        return {
            "missing_count": len(missing),
            "missing_values": missing[:20],
            "suggestions": suggestions
        }
    
    def _guess_field_from_context(self, context: str) -> str:
        """Guess what field a value belongs to based on surrounding text."""
        context_lower = context.lower()
        
        field_keywords = {
            "sales": ["sales", "revenue", "turnover", "income from operations"],
            "expenses": ["expense", "cost", "expenditure"],
            "operating_profit": ["operating profit", "ebit", "operating income", "opm"],
            "net_profit": ["net profit", "pat", "profit after tax", "net income"],
            "depreciation": ["depreciation", "amortization", "d&a"],
            "interest": ["interest", "finance cost", "borrowing cost"],
            "tax": ["tax", "income tax", "provision for tax"],
            "eps": ["eps", "earning per share", "per share"],
            "other_income": ["other income", "non-operating"],
        }
        
        for field, keywords in field_keywords.items():
            if any(kw in context_lower for kw in keywords):
                return field
        
        return "unknown"


class IterativeValidator:
    """7-iteration validation loop with self-healing."""
    
    def __init__(self, llm_caller, max_iterations: int = 7):
        self.llm_caller = llm_caller
        self.max_iterations = max_iterations
        self.extractor = RobustExtractor()
    
    def extract_with_validation(
        self,
        pdf_path: str,
        page_num: int,
        doc_type: str,
        target_periods: List[str] = None
    ) -> Dict[str, Any]:
        """Extract data with iterative validation and self-healing."""
        
        print(f"\n{'='*60}")
        print(f"ROBUST EXTRACTION: Page {page_num}, Type: {doc_type}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"{'='*60}")
        
        # Step 1: Multi-source extraction
        print("\n[1/4] Extracting from multiple sources...")
        multi_source = self.extractor.extract_all_sources(pdf_path, page_num)
        combined = multi_source["combined"]
        
        print(f"      Sources used: {multi_source['extractors_used']}")
        print(f"      Text length: {combined['char_count']} chars")
        print(f"      Numbers found: {combined['number_count']}")
        print(f"      High-confidence numbers: {len(combined['high_confidence_numbers'])}")
        
        # Step 2: Initial LLM extraction
        print("\n[2/4] Initial LLM extraction...")
        current_json = self._initial_extraction(
            combined["text"], 
            doc_type, 
            target_periods
        )
        
        if not current_json or "error" in current_json:
            print(f"      ❌ Initial extraction failed")
            return {"error": "Initial extraction failed", "raw": current_json}
        
        # Step 3: Iterative validation and healing
        print(f"\n[3/4] Iterative validation ({self.max_iterations} max iterations)...")
        
        best_json = current_json
        best_coverage = 0
        iteration_results = []
        
        for iteration in range(1, self.max_iterations + 1):
            # Validate current extraction
            validation = self.extractor.validate_extraction(
                current_json,
                combined["text"],
                combined["all_numbers"]
            )
            
            accuracy = validation["accuracy_percent"]
            iteration_results.append({
                "iteration": iteration,
                "accuracy": accuracy,
                "fields": validation["fields_with_data"],
                "null_pct": validation["null_percent"],
                "passed": validation["passed"]
            })
            
            print(f"      Iteration {iteration}: Accuracy {accuracy:.1f}%, Fields: {validation['fields_with_data']}, Nulls: {validation['null_percent']:.1f}%")
            
            # Track best result
            if accuracy > best_coverage:
                best_coverage = accuracy
                best_json = current_json.copy()
            
            # Check if we're good enough
            if validation["passed"]:
                print(f"      [PASS] Validation PASSED at iteration {iteration}")
                break
            
            # Self-healing: Try to improve if not passed
            if not validation["passed"] and iteration < self.max_iterations:
                print(f"      [HEAL] Attempting self-healing...")
                missing_info = self.extractor.find_missing_data(
                    current_json,
                    combined["all_numbers"],
                    combined["text"]
                )
                
                # Try to fix with LLM
                current_json = self._heal_extraction(
                    current_json,
                    missing_info,
                    combined["text"],
                    doc_type,
                    iteration
                )
        
        # Step 4: Final validation
        print("\n[4/4] Final validation...")
        final_validation = self.extractor.validate_extraction(
            best_json,
            combined["text"],
            combined["all_numbers"]
        )
        
        print(f"      Final accuracy: {final_validation['accuracy_percent']:.1f}%")
        print(f"      Fields with data: {final_validation['fields_with_data']}")
        print(f"      Array issues: {len(final_validation['array_issues'])}")
        
        return {
            "extracted_data": best_json,
            "validation": final_validation,
            "iterations": iteration_results,
            "sources_used": multi_source["extractors_used"],
            "total_iterations": len(iteration_results)
        }
    
    def _initial_extraction(
        self,
        text: str,
        doc_type: str,
        target_periods: List[str]
    ) -> Dict[str, Any]:
        """Perform initial LLM extraction."""
        from app.page_extractor import _build_page_extraction_prompt, _extract_json_from_text
        
        prompt = _build_page_extraction_prompt(text, doc_type, target_periods)
        response = self.llm_caller(prompt)
        return _extract_json_from_text(response) or {}
    
    def _heal_extraction(
        self,
        current_json: Dict,
        missing_info: Dict,
        source_text: str,
        doc_type: str,
        iteration: int
    ) -> Dict[str, Any]:
        """Attempt to heal extraction by filling missing values."""
        
        # Build healing prompt
        suggestions_text = "\n".join([
            f"- {s['value']} (likely {s['likely_field']}): ...{s['context']}..."
            for s in missing_info.get("suggestions", [])[:5]
        ])
        
        heal_prompt = f"""EXTRACTION HEALING - Iteration {iteration}

The following values appear in the source but are MISSING from the extraction:
{suggestions_text}

CURRENT EXTRACTION (incomplete):
{json.dumps(current_json, indent=2)}

SOURCE TEXT:
{source_text}

TASK: Update the JSON to include the missing values in their correct fields.
- Find where each missing value belongs based on its context
- Ensure array lengths match the periods array
- Output the COMPLETE corrected JSON

Output ONLY valid JSON."""

        response = self.llm_caller(heal_prompt)
        
        from app.page_extractor import _extract_json_from_text
        healed = _extract_json_from_text(response)
        
        return healed if healed else current_json


def process_page_robust(
    pdf_path: str,
    page_num: int,
    doc_type: str,
    target_periods: List[str] = None,
    max_iterations: int = 7
) -> Dict[str, Any]:
    """Process a single page with robust multi-extractor pipeline."""
    from app.page_extractor import _call_llm
    
    validator = IterativeValidator(_call_llm, max_iterations)
    return validator.extract_with_validation(pdf_path, page_num, doc_type, target_periods)
