"""
Unit tests for the audit module.

Tests the data extraction audit functionality that verifies
all data from source Excel/CSV files is present in extracted JSON.
"""

import os
import sys
import json
import tempfile
import pytest
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.audit import (
    AuditResult,
    audit_extraction,
    _extract_numeric_from_value,
    _analyze_source_excel,
    _analyze_source_csv,
    _analyze_extracted_json,
)


class TestExtractNumericFromValue:
    """Tests for _extract_numeric_from_value function."""
    
    def test_integer_value(self):
        """Test extraction of integer values."""
        assert _extract_numeric_from_value(42) == 42.0
    
    def test_float_value(self):
        """Test extraction of float values."""
        assert _extract_numeric_from_value(3.14) == 3.14
    
    def test_string_numeric(self):
        """Test extraction of numeric strings."""
        assert _extract_numeric_from_value("123.45") == 123.45
    
    def test_string_with_commas(self):
        """Test extraction of numeric strings with commas."""
        assert _extract_numeric_from_value("1,234,567.89") == 1234567.89
    
    def test_percentage_string(self):
        """Test extraction of percentage strings."""
        assert _extract_numeric_from_value("27%") == 27.0
        assert _extract_numeric_from_value("12.5%") == 12.5
    
    def test_none_value(self):
        """Test handling of None values."""
        assert _extract_numeric_from_value(None) is None
    
    def test_non_numeric_string(self):
        """Test handling of non-numeric strings."""
        assert _extract_numeric_from_value("hello") is None
        assert _extract_numeric_from_value("N/A") is None


class TestAuditResult:
    """Tests for AuditResult dataclass."""
    
    def test_audit_result_creation(self):
        """Test creating an AuditResult object."""
        result = AuditResult(
            passed=True,
            source_file="test.xlsx",
            json_file="test.json",
            source_row_count=100,
            json_row_count=100,
        )
        assert result.passed is True
        assert result.source_file == "test.xlsx"
        assert result.source_row_count == 100
    
    def test_audit_result_to_dict(self):
        """Test converting AuditResult to dictionary."""
        result = AuditResult(
            passed=True,
            source_file="test.xlsx",
            json_file="test.json",
            source_row_count=50,
            json_row_count=50,
            source_numeric_sum=1000.0,
            json_numeric_sum=1000.0,
        )
        result_dict = result.to_dict()
        
        assert result_dict["passed"] is True
        assert result_dict["source_file"] == "test.xlsx"
        assert result_dict["summary"]["source_rows"] == 50
        assert result_dict["summary"]["row_match"] is True


class TestAnalyzeSourceExcel:
    """Tests for _analyze_source_excel function."""
    
    def test_analyze_excel_file(self, tmp_path):
        """Test analyzing an Excel file."""
        # Create a temporary Excel file using pytest tmp_path
        temp_path = tmp_path / "test_data.xlsx"
        
        # Create test data
        df = pd.DataFrame({
            "Name": ["Alice", "Bob", "Charlie"],
            "Sales": [100.0, 200.0, 300.0],
            "Quantity": [10, 20, 30],
        })
        df.to_excel(temp_path, index=False)
        
        # Analyze
        result = _analyze_source_excel(str(temp_path))
        
        assert result["sheet_count"] == 1
        assert result["total_rows"] == 3
        assert result["total_numeric_sum"] == 660.0  # 100+200+300+10+20+30
        assert result["total_numeric_count"] == 6


class TestAnalyzeSourceCsv:
    """Tests for _analyze_source_csv function."""
    
    def test_analyze_csv_file(self):
        """Test analyzing a CSV file."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="") as f:
            f.write("Name,Value\n")
            f.write("Item1,100\n")
            f.write("Item2,200\n")
            f.write("Item3,300\n")
            temp_path = f.name
        
        try:
            result = _analyze_source_csv(temp_path)
            
            assert result["sheet_count"] == 1
            assert result["total_rows"] == 3
            assert result["total_numeric_sum"] == 600.0
            assert result["total_numeric_count"] == 3
        finally:
            os.unlink(temp_path)


class TestAnalyzeExtractedJson:
    """Tests for _analyze_extracted_json function."""
    
    def test_analyze_json_with_sheets(self):
        """Test analyzing JSON with sheets structure."""
        json_data = {
            "sheets": {
                "Sheet1": {
                    "data": [
                        {"col1": 10, "col2": 20},
                        {"col1": 30, "col2": 40},
                    ],
                    "columns": ["col1", "col2"],
                }
            }
        }
        
        result = _analyze_extracted_json(json_data)
        
        assert result["sheet_count"] == 1
        assert result["total_rows"] == 2
        assert result["total_numeric_sum"] == 100.0  # 10+20+30+40
    
    def test_analyze_json_with_extracted_data(self):
        """Test analyzing JSON with extracted_data wrapper."""
        json_data = {
            "extracted_data": {
                "sheets": {
                    "Data": {
                        "data": [{"value": 50}],
                        "columns": ["value"],
                    }
                }
            }
        }
        
        result = _analyze_extracted_json(json_data)
        
        assert result["total_numeric_sum"] == 50.0


class TestAuditExtraction:
    """Tests for audit_extraction function."""
    
    def test_audit_matching_data(self, tmp_path):
        """Test audit with matching source and JSON data."""
        # Create temporary Excel file using pytest tmp_path
        temp_excel = tmp_path / "test_audit.xlsx"
        
        df = pd.DataFrame({
            "Item": ["A", "B"],
            "Value": [100.0, 200.0],
        })
        df.to_excel(temp_excel, index=False)
        
        # Create matching JSON data
        json_data = {
            "sheets": {
                "Sheet1": {
                    "data": [
                        {"Item": "A", "Value": 100.0},
                        {"Item": "B", "Value": 200.0},
                    ],
                    "columns": ["Item", "Value"],
                    "row_count": 2,
                }
            }
        }
        
        result = audit_extraction(str(temp_excel), json_data=json_data)
        
        # Should pass - data matches
        assert result.source_row_count == 2
        assert result.source_numeric_count >= 2
    
    def test_audit_with_missing_data(self, tmp_path):
        """Test audit detects missing data."""
        # Create temporary Excel file with more data
        temp_excel = tmp_path / "test_audit_missing.xlsx"
        
        df = pd.DataFrame({
            "Item": ["A", "B", "C", "D"],
            "Value": [100.0, 200.0, 300.0, 400.0],
        })
        df.to_excel(temp_excel, index=False)
        
        # Create JSON data with less data (missing items)
        json_data = {
            "sheets": {
                "Sheet1": {
                    "data": [
                        {"Item": "A", "Value": 100.0},
                    ],
                    "columns": ["Item", "Value"],
                    "row_count": 1,
                }
            }
        }
        
        result = audit_extraction(str(temp_excel), json_data=json_data)
        
        # Should detect the difference
        assert result.source_row_count == 4
        assert result.json_row_count == 1
        # Numeric sums should differ
        assert result.source_numeric_sum > result.json_numeric_sum


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
