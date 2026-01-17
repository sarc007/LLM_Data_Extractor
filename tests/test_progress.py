"""
Tests for progress tracking module.
"""
import pytest
import time
import asyncio
from app.progress import (
    create_tracker, get_tracker, update_progress, 
    cleanup_tracker, generate_progress_events, ProgressTracker
)


class TestProgressTracker:
    """Test ProgressTracker class."""
    
    def test_create_tracker(self):
        """Test creating a new progress tracker."""
        tracker = create_tracker("test-job-1")
        assert tracker is not None
        assert tracker.job_id == "test-job-1"
        assert tracker.status == "starting"
        assert tracker.completed is False
        cleanup_tracker("test-job-1")
    
    def test_get_tracker(self):
        """Test retrieving an existing tracker."""
        create_tracker("test-job-2")
        tracker = get_tracker("test-job-2")
        assert tracker is not None
        assert tracker.job_id == "test-job-2"
        cleanup_tracker("test-job-2")
    
    def test_get_nonexistent_tracker(self):
        """Test retrieving a non-existent tracker returns None."""
        tracker = get_tracker("nonexistent-job")
        assert tracker is None
    
    def test_update_progress_status(self):
        """Test updating progress status."""
        create_tracker("test-job-3")
        update_progress("test-job-3", status="processing", message="Working...")
        
        tracker = get_tracker("test-job-3")
        assert tracker.status == "processing"
        assert tracker.message == "Working..."
        cleanup_tracker("test-job-3")
    
    def test_update_progress_page_iteration(self):
        """Test updating page and iteration progress."""
        create_tracker("test-job-4")
        update_progress(
            "test-job-4",
            total_pages=10,
            total_iterations=5,
            current_page=3,
            current_iteration=2
        )
        
        tracker = get_tracker("test-job-4")
        assert tracker.total_pages == 10
        assert tracker.total_iterations == 5
        assert tracker.current_page == 3
        assert tracker.current_iteration == 2
        cleanup_tracker("test-job-4")
    
    def test_progress_percent_calculation(self):
        """Test progress percentage calculation."""
        tracker = create_tracker("test-job-5")
        
        # Initial state - 0%
        assert tracker._calc_progress() == 0
        
        # Set total pages
        update_progress("test-job-5", total_pages=10, total_iterations=5)
        
        # Page 1, iteration 1 = 1/50 = 2%
        update_progress("test-job-5", current_page=1, current_iteration=1)
        tracker = get_tracker("test-job-5")
        assert tracker._calc_progress() == 2  # (0*5 + 1) / 50 = 2%
        
        # Page 5, iteration 3 = 23/50 = 46%
        update_progress("test-job-5", current_page=5, current_iteration=3)
        tracker = get_tracker("test-job-5")
        assert tracker._calc_progress() == 46  # (4*5 + 3) / 50 = 46%
        
        cleanup_tracker("test-job-5")
    
    def test_update_completed(self):
        """Test marking job as completed."""
        create_tracker("test-job-6")
        update_progress("test-job-6", completed=True, message="Done!")
        
        tracker = get_tracker("test-job-6")
        assert tracker.completed is True
        assert tracker.message == "Done!"
        cleanup_tracker("test-job-6")
    
    def test_update_error(self):
        """Test marking job with error."""
        create_tracker("test-job-7")
        update_progress("test-job-7", error="Something went wrong")
        
        tracker = get_tracker("test-job-7")
        assert tracker.error == "Something went wrong"
        cleanup_tracker("test-job-7")
    
    def test_to_dict(self):
        """Test tracker serialization to dict."""
        create_tracker("test-job-8")
        update_progress(
            "test-job-8",
            status="processing",
            message="Page 2 processing",
            total_pages=5,
            current_page=2,
            current_iteration=3,
            total_iterations=5
        )
        
        tracker = get_tracker("test-job-8")
        data = tracker.to_dict()
        
        assert data["job_id"] == "test-job-8"
        assert data["status"] == "processing"
        assert data["message"] == "Page 2 processing"
        assert data["total_pages"] == 5
        assert data["current_page"] == 2
        assert data["current_iteration"] == 3
        assert "progress_percent" in data
        cleanup_tracker("test-job-8")
    
    def test_cleanup_tracker(self):
        """Test cleanup removes tracker."""
        create_tracker("test-job-9")
        assert get_tracker("test-job-9") is not None
        
        cleanup_tracker("test-job-9")
        assert get_tracker("test-job-9") is None


class TestSSEGenerator:
    """Test SSE event generation (sync tests only - async requires pytest-asyncio)."""
    
    def test_generate_progress_events_is_async_generator(self):
        """Test that generate_progress_events returns an async generator."""
        import inspect
        assert inspect.isasyncgenfunction(generate_progress_events)


class TestIntegration:
    """Integration tests for progress tracking."""
    
    def test_full_progress_flow(self):
        """Test complete progress tracking flow."""
        job_id = "integration-test-1"
        
        # Create tracker
        tracker = create_tracker(job_id)
        assert tracker.status == "starting"
        
        # Simulate OCR extraction
        update_progress(job_id, status="ocr_extraction", message="Extracting pages...")
        tracker = get_tracker(job_id)
        assert tracker.status == "ocr_extraction"
        
        # Simulate finding pages
        update_progress(job_id, total_pages=5, total_iterations=5)
        tracker = get_tracker(job_id)
        assert tracker.total_pages == 5
        
        # Simulate processing pages
        for page in range(1, 6):
            for iteration in range(1, 6):
                update_progress(
                    job_id,
                    current_page=page,
                    current_iteration=iteration,
                    message=f"Page {page}: Iteration {iteration}/5"
                )
        
        tracker = get_tracker(job_id)
        assert tracker.current_page == 5
        assert tracker.current_iteration == 5
        
        # Mark complete
        update_progress(job_id, completed=True, status="complete")
        tracker = get_tracker(job_id)
        assert tracker.completed is True
        
        # Cleanup
        cleanup_tracker(job_id)
        assert get_tracker(job_id) is None
