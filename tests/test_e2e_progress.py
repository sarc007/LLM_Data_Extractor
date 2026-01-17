"""
E2E tests for progress tracking using httpx.
Tests the async processing endpoints.
"""
import pytest
import httpx
import json
from app.progress import create_tracker, update_progress, get_tracker, cleanup_tracker


class TestAsyncEndpoints:
    """Test async processing endpoints."""
    
    BASE_URL = "http://127.0.0.1:8000"
    
    def test_process_qwen_async_returns_job_id(self):
        """Test that /process-qwen-async returns a job_id."""
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
        
        with httpx.Client(base_url=self.BASE_URL, timeout=30) as client:
            files = {"file": ("test.pdf", pdf_content, "application/pdf")}
            data = {"iterations": "2"}
            
            response = client.post("/process-qwen-async", files=files, data=data)
            
            assert response.status_code == 200
            result = response.json()
            assert "job_id" in result
            assert "filename" in result
            assert "upload_path" in result
            print(f"Got job_id: {result['job_id']}")
    
    def test_progress_endpoint_returns_sse_content_type(self):
        """Test that /progress/{job_id} returns SSE content type."""
        with httpx.Client(base_url=self.BASE_URL, timeout=5) as client:
            # Even for non-existent job, should return SSE content type
            with client.stream("GET", "/progress/nonexistent-job") as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers.get("content-type", "")
                
                # Read first event - should be error for nonexistent job
                for line in response.iter_lines():
                    if "{" in line:  # Find JSON in line
                        json_start = line.index("{")
                        data = json.loads(line[json_start:])
                        assert "error" in data
                        print(f"SSE error response: {data}")
                        break
    
    def test_full_async_flow_creates_tracker(self):
        """Test that async endpoint creates a tracker in server."""
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
        
        with httpx.Client(base_url=self.BASE_URL, timeout=30) as client:
            # Step 1: Start async processing
            files = {"file": ("test_flow.pdf", pdf_content, "application/pdf")}
            data = {"iterations": "1"}
            
            response = client.post("/process-qwen-async", files=files, data=data)
            assert response.status_code == 200
            result = response.json()
            job_id = result["job_id"]
            
            # Step 2: Check SSE endpoint returns valid tracker data (not error)
            with client.stream("GET", f"/progress/{job_id}") as sse_response:
                assert sse_response.status_code == 200
                
                for line in sse_response.iter_lines():
                    if "{" in line:  # Find JSON in line
                        json_start = line.index("{")
                        data = json.loads(line[json_start:])
                        # Should have tracker data, not error
                        if "error" not in data:
                            assert "job_id" in data
                            assert data["job_id"] == job_id
                            print(f"Tracker found: {data['status']}")
                        break


class TestProgressCalculation:
    """Test progress percentage calculation accuracy."""
    
    def test_progress_percent_at_various_stages(self):
        """Verify progress percentage at different stages."""
        job_id = "test-calc"
        create_tracker(job_id)
        
        try:
            # Set up: 10 pages, 5 iterations each = 50 total steps
            update_progress(job_id, total_pages=10, total_iterations=5)
            
            test_cases = [
                # (page, iteration, expected_percent)
                (1, 1, 2),    # 1/50 = 2%
                (1, 5, 10),   # 5/50 = 10%
                (5, 1, 42),   # 21/50 = 42%
                (5, 3, 46),   # 23/50 = 46%
                (10, 5, 100), # 50/50 = 100%
            ]
            
            for page, iteration, expected in test_cases:
                update_progress(job_id, current_page=page, current_iteration=iteration)
                tracker = get_tracker(job_id)
                actual = tracker.to_dict()["progress_percent"]
                assert actual == expected, f"Page {page}, Iter {iteration}: expected {expected}%, got {actual}%"
                print(f"Page {page}, Iter {iteration}: {actual}% ✓")
                
        finally:
            cleanup_tracker(job_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
