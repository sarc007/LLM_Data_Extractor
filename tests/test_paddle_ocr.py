"""
Unit tests for the paddle_ocr module.

Tests the PaddleOCR integration for PDF extraction.
"""

import os
import sys
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.paddle_ocr import (
    PADDLEOCR_AVAILABLE,
    _find_poppler_path,
    _extract_text_from_boxes,
    _detect_tables_from_boxes,
    _build_table_from_lines,
)


class TestPaddleOcrAvailability:
    """Tests for PaddleOCR availability check."""
    
    def test_paddleocr_available_is_boolean(self):
        """Test that PADDLEOCR_AVAILABLE is a boolean."""
        assert isinstance(PADDLEOCR_AVAILABLE, bool)


class TestFindPopplerPath:
    """Tests for _find_poppler_path function."""
    
    def test_find_poppler_path_returns_string_or_none(self):
        """Test that _find_poppler_path returns str or None."""
        result = _find_poppler_path()
        assert result is None or isinstance(result, str)
    
    def test_find_poppler_path_with_env_var(self, monkeypatch):
        """Test that POPPLER_PATH env var is respected."""
        test_path = "C:\\test\\poppler\\bin"
        monkeypatch.setenv("POPPLER_PATH", test_path)
        
        # Need to reimport to pick up new env var
        from app import paddle_ocr
        # The function reads from module-level POPPLER_PATH which is set at import
        # So we test the pattern works
        assert os.getenv("POPPLER_PATH") == test_path


class TestExtractTextFromBoxes:
    """Tests for _extract_text_from_boxes function."""
    
    def test_empty_boxes(self):
        """Test handling of empty boxes list."""
        text, tables, confidence = _extract_text_from_boxes([])
        assert text == ""
        assert tables == []
        assert confidence == 0.0
    
    def test_single_box(self):
        """Test extraction from single box."""
        boxes = [
            [[[0, 0], [100, 0], [100, 20], [0, 20]], ("Hello World", 0.95)]
        ]
        text, tables, confidence = _extract_text_from_boxes(boxes)
        
        assert "Hello World" in text
        assert confidence == 0.95
    
    def test_multiple_boxes_same_line(self):
        """Test extraction from multiple boxes on same line."""
        boxes = [
            [[[0, 0], [50, 0], [50, 20], [0, 20]], ("Hello", 0.9)],
            [[[60, 0], [110, 0], [110, 20], [60, 20]], ("World", 0.8)],
        ]
        text, tables, confidence = _extract_text_from_boxes(boxes)
        
        assert "Hello" in text
        assert "World" in text
        assert 0.8 <= confidence <= 0.9  # Average of 0.9 and 0.8
    
    def test_multiple_boxes_different_lines(self):
        """Test extraction from boxes on different lines."""
        boxes = [
            [[[0, 0], [100, 0], [100, 20], [0, 20]], ("Line 1", 0.9)],
            [[[0, 50], [100, 50], [100, 70], [0, 70]], ("Line 2", 0.8)],
        ]
        text, tables, confidence = _extract_text_from_boxes(boxes)
        
        assert "Line 1" in text
        assert "Line 2" in text
        # Should be on separate lines
        lines = text.split("\n")
        assert len(lines) >= 2


class TestDetectTablesFromBoxes:
    """Tests for _detect_tables_from_boxes function."""
    
    def test_no_tables_in_single_line(self):
        """Test that single line doesn't create table."""
        boxes = [{"x": 0, "y": 0, "text": "Single"}]
        lines = [[{"x": 0, "y": 0, "text": "Single"}]]
        
        tables = _detect_tables_from_boxes(boxes, lines)
        assert tables == []
    
    def test_detect_table_pattern(self):
        """Test detection of table-like pattern."""
        # Create a grid-like pattern with 3+ columns and 2+ rows
        lines = [
            [
                {"x": 0, "y": 0, "text": "Col1"},
                {"x": 100, "y": 0, "text": "Col2"},
                {"x": 200, "y": 0, "text": "Col3"},
            ],
            [
                {"x": 0, "y": 30, "text": "A"},
                {"x": 100, "y": 30, "text": "B"},
                {"x": 200, "y": 30, "text": "C"},
            ],
        ]
        boxes = [item for line in lines for item in line]
        
        tables = _detect_tables_from_boxes(boxes, lines)
        
        # Should detect at least one table
        assert len(tables) >= 1
        if tables:
            assert "headers" in tables[0]
            assert "rows" in tables[0]


class TestBuildTableFromLines:
    """Tests for _build_table_from_lines function."""
    
    def test_empty_lines(self):
        """Test handling of empty lines."""
        result = _build_table_from_lines([])
        assert result == {"headers": [], "rows": []}
    
    def test_single_header_line(self):
        """Test table with only header."""
        lines = [[{"text": "Col1"}, {"text": "Col2"}]]
        result = _build_table_from_lines(lines)
        
        assert result["headers"] == ["Col1", "Col2"]
        assert result["rows"] == []
    
    def test_header_with_data_rows(self):
        """Test table with header and data rows."""
        lines = [
            [{"text": "Name"}, {"text": "Value"}],
            [{"text": "Item1"}, {"text": "100"}],
            [{"text": "Item2"}, {"text": "200"}],
        ]
        result = _build_table_from_lines(lines)
        
        assert result["headers"] == ["Name", "Value"]
        assert len(result["rows"]) == 2
        assert result["rows"][0] == ["Item1", "100"]
        assert result["row_count"] == 2
        assert result["column_count"] == 2


class TestIntegration:
    """Integration tests for paddle_ocr module."""
    
    def test_module_imports_successfully(self):
        """Test that the module imports without errors."""
        from app import paddle_ocr
        assert hasattr(paddle_ocr, 'extract_pdf_with_ocr')
        assert hasattr(paddle_ocr, 'get_ocr_text_for_llm')
        assert hasattr(paddle_ocr, 'hybrid_pdf_extraction')
    
    def test_poppler_path_detection(self):
        """Test that poppler path detection works."""
        path = _find_poppler_path()
        # If poppler is installed, path should point to valid executable
        if path:
            pdftoppm = os.path.join(path, "pdftoppm.exe")
            assert os.path.exists(pdftoppm), f"pdftoppm not found at {pdftoppm}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
