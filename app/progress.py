"""
Progress tracking module for real-time SSE updates.
"""
import asyncio
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading

# Global progress store
_progress_store: Dict[str, 'ProgressTracker'] = {}
_lock = threading.Lock()


@dataclass
class ProgressTracker:
    """Track progress for a processing job."""
    job_id: str
    total_pages: int = 0
    current_page: int = 0
    current_iteration: int = 0
    total_iterations: int = 5
    status: str = "starting"
    message: str = ""
    page_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    error: Optional[str] = None
    document_types_found: list = field(default_factory=list)
    current_doc_type: str = ""
    audit_status: str = ""
    output_folder: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "total_pages": self.total_pages,
            "current_page": self.current_page,
            "current_iteration": self.current_iteration,
            "total_iterations": self.total_iterations,
            "status": self.status,
            "message": self.message,
            "completed": self.completed,
            "error": self.error,
            "progress_percent": self._calc_progress(),
            "document_types_found": self.document_types_found,
            "current_doc_type": self.current_doc_type,
            "audit_status": self.audit_status,
            "pages_completed": len(self.page_results),
            "output_folder": self.output_folder,
        }
    
    def _calc_progress(self) -> int:
        if self.total_pages == 0:
            return 0
        # Each page has total_iterations steps
        total_steps = self.total_pages * self.total_iterations
        completed_steps = (self.current_page - 1) * self.total_iterations + self.current_iteration
        return min(100, int((completed_steps / total_steps) * 100))


def create_tracker(job_id: str) -> ProgressTracker:
    """Create a new progress tracker."""
    with _lock:
        tracker = ProgressTracker(job_id=job_id)
        _progress_store[job_id] = tracker
        return tracker


def get_tracker(job_id: str) -> Optional[ProgressTracker]:
    """Get progress tracker by job ID."""
    with _lock:
        return _progress_store.get(job_id)


def update_progress(
    job_id: str,
    status: str = None,
    message: str = None,
    current_page: int = None,
    current_iteration: int = None,
    total_pages: int = None,
    total_iterations: int = None,
    page_json: Dict[str, Any] = None,
    completed: bool = None,
    error: str = None,
    output_folder: str = None
):
    """Update progress for a job."""
    with _lock:
        tracker = _progress_store.get(job_id)
        if not tracker:
            return
        
        if status is not None:
            tracker.status = status
        if message is not None:
            tracker.message = message
        if current_page is not None:
            tracker.current_page = current_page
        if current_iteration is not None:
            tracker.current_iteration = current_iteration
        if total_pages is not None:
            tracker.total_pages = total_pages
        if total_iterations is not None:
            tracker.total_iterations = total_iterations
        if page_json is not None and tracker.current_page > 0:
            tracker.page_results[tracker.current_page] = page_json
        if completed is not None:
            tracker.completed = completed
        if error is not None:
            tracker.error = error
        if output_folder is not None:
            tracker.output_folder = output_folder


def cleanup_tracker(job_id: str):
    """Remove tracker after job completes."""
    with _lock:
        if job_id in _progress_store:
            del _progress_store[job_id]


async def generate_progress_events(job_id: str):
    """Generate SSE events for a job."""
    while True:
        tracker = get_tracker(job_id)
        if not tracker:
            yield {"data": json.dumps({'error': 'Job not found'})}
            break
        
        yield {"data": json.dumps(tracker.to_dict())}
        
        if tracker.completed or tracker.error:
            break
        
        await asyncio.sleep(0.5)  # Update every 500ms
