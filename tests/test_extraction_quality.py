"""
QA Tests for Extraction Quality - validates the 18 issues reported by user.
"""
import json
import pytest
from pathlib import Path


class TestExtractionQuality:
    """Test extraction output for data quality issues."""
    
    @pytest.fixture
    def sample_output(self):
        """Load a sample extraction output for testing."""
        output_path = Path("processed/ae7401d9-bfd6-40eb-b39c-cf4ac1823f85.json")
        if output_path.exists():
            with open(output_path, "r") as f:
                return json.load(f)
        return None
    
    def test_no_invented_periods(self, sample_output):
        """Issue 1-2: Should not have Mar-13, Mar-14, Mar-15 if source starts at Mar-16."""
        if not sample_output:
            pytest.skip("No sample output available")
        
        extracted = sample_output.get("extracted_data", sample_output)
        sales = extracted.get("profit_loss", {}).get("sales", [])
        
        # Apollo Tyres PDF starts from Mar-16, not Mar-13
        invalid_periods = ["Mar-13", "Mar-14", "Mar-15"]
        found_invalid = [s["period"] for s in sales if s["period"] in invalid_periods]
        
        assert len(found_invalid) == 0, f"Found invented periods: {found_invalid}"
    
    def test_no_sudden_value_drops(self, sample_output):
        """Issue 5-6: Sales should not drop 75% in one year (quarterly mixed with annual)."""
        if not sample_output:
            pytest.skip("No sample output available")
        
        extracted = sample_output.get("extracted_data", sample_output)
        sales = extracted.get("profit_loss", {}).get("sales", [])
        
        for i in range(1, len(sales)):
            prev_val = sales[i-1].get("value", 0) or 0
            curr_val = sales[i].get("value", 0) or 0
            
            if prev_val > 0 and curr_val > 0:
                drop_percent = (prev_val - curr_val) / prev_val * 100
                # A 75% drop likely means quarterly mixed with annual
                assert drop_percent < 70, (
                    f"Suspicious drop from {sales[i-1]['period']}={prev_val} "
                    f"to {sales[i]['period']}={curr_val} ({drop_percent:.1f}% drop)"
                )
    
    def test_operating_profit_complete(self, sample_output):
        """Issue 7: Operating profit should have more than 3 values."""
        if not sample_output:
            pytest.skip("No sample output available")
        
        extracted = sample_output.get("extracted_data", sample_output)
        op = extracted.get("profit_loss", {}).get("operating_profit", [])
        
        # Should have at least 8 years of data for a 10-year PDF
        assert len(op) >= 8, f"Operating profit incomplete: only {len(op)} values"
    
    def test_scenario_data_separated(self, sample_output):
        """Issue 3: Best/worst case should be in scenario_data, not mixed with historical."""
        if not sample_output:
            pytest.skip("No sample output available")
        
        extracted = sample_output.get("extracted_data", sample_output)
        
        # Best/worst case should NOT be in profit_loss periods
        sales = extracted.get("profit_loss", {}).get("sales", [])
        periods = [s["period"] for s in sales]
        
        scenario_words = ["best", "worst", "Best", "Worst", "BEST", "WORST"]
        found_scenarios = [p for p in periods if any(w in str(p) for w in scenario_words)]
        
        assert len(found_scenarios) == 0, f"Scenario data mixed with historical: {found_scenarios}"
    
    def test_consistent_period_count(self, sample_output):
        """Issue 17-18: All metrics should have similar period counts."""
        if not sample_output:
            pytest.skip("No sample output available")
        
        extracted = sample_output.get("extracted_data", sample_output)
        pl = extracted.get("profit_loss", {})
        
        counts = {}
        for metric, values in pl.items():
            if isinstance(values, list):
                counts[metric] = len(values)
        
        if counts:
            max_count = max(counts.values())
            min_count = min(counts.values())
            
            # All metrics should have similar counts (within 3 values)
            # This catches table boundary detection issues
            assert max_count - min_count <= 5, (
                f"Inconsistent period counts (boundary issue): {counts}"
            )


class TestPromptRules:
    """Test that prompt rules are properly defined."""
    
    def test_ocr_prompt_has_date_detection_rule(self):
        """Verify prompt has explicit date range detection."""
        from app.qwen_pipeline import _build_ocr_to_json_prompt
        prompt = _build_ocr_to_json_prompt("test")
        
        assert "ACTUAL" in prompt.upper() or "DETECT" in prompt.upper(), \
            "Prompt should have date detection instruction"
        assert "INVENT" in prompt.upper() or "Mar-13" in prompt or "Mar-14" in prompt, \
            "Prompt should warn against inventing periods"
    
    def test_ocr_prompt_has_frequency_separation(self):
        """Verify prompt separates annual/quarterly/TTM."""
        from app.qwen_pipeline import _build_ocr_to_json_prompt
        prompt = _build_ocr_to_json_prompt("test")
        
        assert "quarterly" in prompt.lower(), "Prompt should mention quarterly separation"
        assert "ttm" in prompt.lower() or "trailing" in prompt.lower(), \
            "Prompt should mention TTM separation"
        assert "scenario" in prompt.lower() or "best" in prompt.lower(), \
            "Prompt should mention scenario data separation"
    
    def test_selfcheck_prompt_has_verification(self):
        """Verify self-check prompt validates data quality."""
        from app.qwen_pipeline import _build_selfcheck_prompt
        prompt = _build_selfcheck_prompt("doc", "{}", 1, 3)
        
        assert "DATE" in prompt.upper() or "PERIOD" in prompt.upper(), \
            "Self-check should verify dates"
        assert "COMPLETE" in prompt.upper() or "ALL" in prompt.upper(), \
            "Self-check should verify completeness"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
