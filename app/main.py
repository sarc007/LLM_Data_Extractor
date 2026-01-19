import os
import uuid
import json
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, Form, Depends, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from sse_starlette.sse import EventSourceResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

load_dotenv()

from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse

from .database import Base, engine, get_db
from .models import Upload
from .pipeline import process_document
from .qwen_pipeline import process_document_qwen
from .page_extractor import process_pdf_page_by_page_v2
from .auth import require_login, login_user, AUTH_ENABLED
from .progress import create_tracker, get_tracker, update_progress, generate_progress_events, cleanup_tracker

from sqlalchemy.orm import Session

# Ensure DB tables
Base.metadata.create_all(bind=engine)

# FS dirs
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)

app = FastAPI()

SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db), _=Depends(require_login)):
    # Show last 10 entries
    uploads = db.query(Upload).order_by(Upload.created_at.desc()).limit(10).all()
    return templates.TemplateResponse("index.html", {"request": request, "uploads": uploads, "auth_enabled": AUTH_ENABLED})


@app.post("/process", response_class=HTMLResponse)
async def process_file(
    request: Request,
    file: UploadFile,
    db: Session = Depends(get_db),
    _=Depends(require_login),
):
    # Save uploaded file
    uid = str(uuid.uuid4())
    filename = f"{uid}_{file.filename}"
    upload_path = os.path.join("uploads", filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Check if PDF - use page-by-page extraction for PDFs
    is_pdf = file.filename.lower().endswith('.pdf')
    
    if is_pdf:
        # Use page-by-page extraction for PDFs (creates human-readable folders)
        try:
            import asyncio
            pdf_name = Path(file.filename).stem
            result = await asyncio.to_thread(
                process_pdf_page_by_page_v2,
                pdf_path=upload_path,
                iterations_per_page=3,
                combine_iterations=3,
                use_ocr=True,
                use_robust=True,
            )
            # Get the output folder path
            output_folder = f"{pdf_name}_pages"
            json_path = os.path.join("processed", output_folder, "extraction_result.json")
            json_result = result
            model_used = "page-by-page (robust)"
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    else:
        # Run old pipeline for non-PDFs
        try:
            json_result, model_used = process_document(upload_path, "format_template.json")
        except Exception as e:
            return JSONResponse({"success": False, "error": str(e)}, status_code=500)
        
        # Save JSON to processed/ with UUID for non-PDFs
        json_path = os.path.join("processed", f"{uid}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)

    # Insert into DB
    upload = Upload(
        id=uid,
        filename=file.filename,
        original_path=upload_path,
        json_path=json_path,
        model_used=model_used,
    )
    db.add(upload)
    db.commit()

    # Render result fragment (JSON pretty + tables + download link)
    return templates.TemplateResponse(
        "result_fragment.html",
        {
            "request": request,
            "upload": upload,
            "json_data": json_result,
        },
    )


@app.get("/download/{upload_id}")
async def download_json(upload_id: str, db: Session = Depends(get_db), _=Depends(require_login)):
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        return JSONResponse({"error": "Not found"}, status_code=404)
    # Use original filename for download (human-readable, not UUID)
    original_name = Path(upload.filename).stem if upload.filename else upload_id
    return FileResponse(
        upload.json_path,
        media_type="application/json",
        filename=f"{original_name}_extracted.json"
    )


@app.post("/process-qwen", response_class=HTMLResponse)
async def process_file_qwen(
    request: Request,
    file: UploadFile,
    iterations: int = Form(5),
    convert_pdf: bool = Form(False),
    use_paddleocr: bool = Form(True),  # Default True for better PDF extraction
    ocr_dpi: int = Form(200),
    db: Session = Depends(get_db),
    _=Depends(require_login),
):
    """Process file using Qwen3 480B Cloud with iterative self-checking."""
    # Save uploaded file
    uid = str(uuid.uuid4())
    filename = f"{uid}_{file.filename}"
    upload_path = os.path.join("uploads", filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Auto-enable PaddleOCR for PDFs
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext == ".pdf" and use_paddleocr:
        print(f"[INFO] PDF detected - using PaddleOCR for extraction")

    # Run Qwen pipeline
    try:
        extracted_data, analysis = process_document_qwen(
            upload_path,
            convert_to_pdf_first=convert_pdf,
            iterations=iterations,
            use_paddleocr=use_paddleocr,
            run_audit=True,
            ocr_dpi=ocr_dpi
        )
        
        # Combine results
        json_result = {
            "extracted_data": extracted_data,
            "analysis": analysis,
            "metadata": {
                "source_file": file.filename,
                "model": "qwen3-coder:480b-cloud",
                "iterations": iterations
            }
        }
        model_used = "qwen3-coder:480b-cloud"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

    # Save JSON to processed/
    json_path = os.path.join("processed", f"{uid}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)

    # Insert into DB
    upload = Upload(
        id=uid,
        filename=file.filename,
        original_path=upload_path,
        json_path=json_path,
        model_used=model_used,
    )
    db.add(upload)
    db.commit()

    # Render result fragment
    return templates.TemplateResponse(
        "result_fragment.html",
        {
            "request": request,
            "upload": upload,
            "json_data": json_result,
        },
    )


# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=2)


@app.get("/progress/{job_id}")
async def progress_stream(job_id: str):
    """SSE endpoint for real-time progress updates."""
    return EventSourceResponse(generate_progress_events(job_id))


@app.post("/process-qwen-async")
async def process_file_qwen_async(
    request: Request,
    file: UploadFile,
    iterations: int = Form(5),
    convert_pdf: bool = Form(False),
    use_paddleocr: bool = Form(True),
    ocr_dpi: int = Form(200),
    db: Session = Depends(get_db),
    _=Depends(require_login),
):
    """Start async processing and return job_id for progress tracking."""
    uid = str(uuid.uuid4())
    filename = f"{uid}_{file.filename}"
    upload_path = os.path.join("uploads", filename)
    
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create progress tracker
    create_tracker(uid)
    update_progress(uid, status="starting", message="Processing started...")
    
    # Return job_id immediately for SSE tracking
    return JSONResponse({
        "job_id": uid,
        "filename": file.filename,
        "upload_path": upload_path,
        "iterations": iterations,
        "use_paddleocr": use_paddleocr,
        "ocr_dpi": ocr_dpi
    })


@app.post("/process-qwen-execute/{job_id}")
async def execute_processing(
    job_id: str,
    request: Request,
    db: Session = Depends(get_db),
    _=Depends(require_login),
):
    """Execute the actual processing for a job in background thread."""
    data = await request.json()
    upload_path = data.get("upload_path")
    filename = data.get("filename")
    iterations = data.get("iterations", 5)
    use_paddleocr = data.get("use_paddleocr", True)
    ocr_dpi = data.get("ocr_dpi", 200)
    convert_pdf = data.get("convert_pdf", False)
    
    try:
        # Run in background thread to not block SSE event loop
        import asyncio
        extracted_data, analysis = await asyncio.to_thread(
            process_document_qwen,
            upload_path,
            convert_to_pdf_first=convert_pdf,
            iterations=iterations,
            use_paddleocr=use_paddleocr,
            run_audit=True,
            ocr_dpi=ocr_dpi,
            job_id=job_id  # Pass job_id for progress tracking
        )
        
        json_result = {
            "extracted_data": extracted_data,
            "analysis": analysis,
            "metadata": {
                "source_file": filename,
                "model": "qwen3-coder:480b-cloud",
                "iterations": iterations
            }
        }
        model_used = "qwen3-coder:480b-cloud"
        
        # Save JSON to processed/
        json_path = os.path.join("processed", f"{job_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
        
        # Insert into SQLite DB
        upload = Upload(
            id=job_id,
            filename=filename,
            original_path=upload_path,
            json_path=json_path,
            model_used=model_used,
        )
        db.add(upload)
        db.commit()
        
        # Save to PostgreSQL for querying
        try:
            from app.postgres_engine import get_connection
            from psycopg2.extras import Json
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO source_files (file_name, file_type, company_name, data_type, raw_json)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (file_name) DO UPDATE SET raw_json = EXCLUDED.raw_json, loaded_at = CURRENT_TIMESTAMP
            """, (filename, 'pdf', filename.replace('.pdf', ''), 
                  analysis.get('data_type', {}).get('primary_type', 'Unknown'),
                  Json(json_result)))
            conn.commit()
            cursor.close()
            conn.close()
            print(f"[OK] Saved to PostgreSQL: {filename}")
        except Exception as pg_err:
            print(f"[WARN] PostgreSQL save failed: {pg_err}")
        
        # For Qwen processing, output goes to job_id.json (not a folder)
        update_progress(job_id, completed=True, message="Processing complete!", output_folder="")
        cleanup_tracker(job_id)
        
        return JSONResponse({"success": True, "upload_id": job_id})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        update_progress(job_id, error=str(e), message=f"Error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/process-page-by-page")
async def process_page_by_page(
    request: Request,
    file: UploadFile,
    iterations: int = Form(3),
    use_ocr: bool = Form(True),
    db: Session = Depends(get_db),
    _=Depends(require_login),
):
    """Process PDF page-by-page with document type detection and progress tracking."""
    uid = str(uuid.uuid4())
    filename = f"{uid}_{file.filename}"
    upload_path = os.path.join("uploads", filename)
    
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create progress tracker
    create_tracker(uid)
    update_progress(uid, status="starting", message="Starting page-by-page extraction...")
    
    try:
        # Run page-by-page extraction
        result = await asyncio.to_thread(
            process_pdf_page_by_page_v2,
            pdf_path=upload_path,
            iterations_per_page=iterations,
            combine_iterations=3,
            use_ocr=use_ocr,
            job_id=uid
        )
        
        # Build JSON result
        json_result = {
            "extraction_result": result,
            "metadata": {
                "source_file": file.filename,
                "model": "qwen3-coder:480b-cloud",
                "extraction_method": "page-by-page",
                "total_pages": result.get("total_pages", 0),
                "documents_found": len(result.get("documents", []))
            }
        }
        
        # Save JSON to processed/
        json_path = os.path.join("processed", f"{uid}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
        
        # Insert into DB
        upload = Upload(
            id=uid,
            filename=file.filename,
            original_path=upload_path,
            json_path=json_path,
            model_used="qwen3-coder:480b-cloud (page-by-page)",
        )
        db.add(upload)
        db.commit()
        
        # Get folder name from result
        output_folder = Path(result.get("output_directory", "")).name
        update_progress(uid, completed=True, message="Extraction complete!", output_folder=output_folder)
        
        return JSONResponse({
            "success": True,
            "upload_id": uid,
            "total_pages": result.get("total_pages", 0),
            "documents_found": len(result.get("documents", [])),
            "document_types": result.get("document_types_found", []),
            "output_folder": output_folder
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        update_progress(uid, error=str(e), message=f"Error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    finally:
        cleanup_tracker(uid)


@app.get("/api/page-extractions")
async def list_page_extractions(_=Depends(require_login)):
    """List all page-by-page extraction results from _pages folders."""
    from pathlib import Path
    processed = Path("processed")
    results = []
    
    for folder in processed.iterdir():
        if folder.is_dir() and folder.name.endswith("_pages"):
            result_file = folder / "extraction_result.json"
            if result_file.exists():
                with open(result_file) as f:
                    data = json.load(f)
                
                # Get combined document files
                combined_files = list(folder.glob("combined_*.json"))
                
                results.append({
                    "folder": folder.name,
                    "source_pdf": data.get("source_pdf", ""),
                    "total_pages": data.get("total_pages", 0),
                    "documents_found": len(data.get("documents", [])),
                    "document_types": data.get("document_types_found", []),
                    "combined_files": [f.name for f in combined_files],
                    "timestamp": data.get("timestamp", "")
                })
    
    return JSONResponse({"extractions": results})


@app.get("/api/processed-folders")
async def list_processed_folders(_=Depends(require_login)):
    """List all processed PDF folders with their JSON files."""
    from pathlib import Path
    processed = Path("processed")
    folders = []
    
    for folder in sorted(processed.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if folder.is_dir() and folder.name.endswith("_pages"):
            # Get all JSON files in folder
            json_files = sorted([f.name for f in folder.glob("*.json")])
            
            # Clean display name (remove _pages suffix)
            display_name = folder.name.replace("_pages", "")
            
            folders.append({
                "name": folder.name,
                "display_name": display_name,
                "files": json_files,
                "file_count": len(json_files)
            })
    
    return JSONResponse({"folders": folders})


@app.get("/view-file/{folder_name}/{file_name}", response_class=HTMLResponse)
async def view_processed_file(folder_name: str, file_name: str, request: Request, _=Depends(require_login)):
    """View a specific JSON file from a processed folder."""
    from pathlib import Path
    
    file_path = Path("processed") / folder_name / file_name
    if not file_path.exists():
        return HTMLResponse("File not found", status_code=404)
    
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # Human-readable display name (no UUIDs)
    display_name = folder_name.replace("_pages", "")
    
    # Create a mock upload object for the template
    class MockUpload:
        def __init__(self, folder, filename, display):
            self.id = folder
            self.filename = f"{display} / {filename}"  # Human-readable
            self.json_path = str(file_path)
            self.model_used = "page-by-page extraction"
            self.created_at = ""
    
    upload = MockUpload(folder_name, file_name, display_name)
    
    return templates.TemplateResponse(
        "view_json.html",
        {
            "request": request,
            "upload": upload,
            "json_data": json_data,
            "auth_enabled": AUTH_ENABLED,
            "folder_name": folder_name,
            "file_name": file_name,
            "display_name": display_name,
        },
    )


@app.get("/browse/{folder_name}", response_class=HTMLResponse)
async def browse_folder(folder_name: str, request: Request, _=Depends(require_login)):
    """Browse all files in a processed folder - human-readable view."""
    from pathlib import Path
    
    folder_path = Path("processed") / folder_name
    if not folder_path.exists():
        return HTMLResponse("Folder not found", status_code=404)
    
    # Get all JSON files
    files = sorted([f.name for f in folder_path.glob("*.json")])
    
    # Categorize files
    combined_files = [f for f in files if f.startswith("combined_")]
    page_files = [f for f in files if f.startswith("page_")]
    other_files = [f for f in files if not f.startswith("combined_") and not f.startswith("page_")]
    
    # Human-readable display name
    display_name = folder_name.replace("_pages", "")
    
    # Build HTML response
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{display_name} - Extracted Files</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 p-8">
        <div class="max-w-4xl mx-auto">
            <div class="bg-white rounded-xl shadow p-6">
                <div class="flex justify-between items-center mb-6">
                    <h1 class="text-2xl font-bold">📂 {display_name}</h1>
                    <a href="/" class="text-blue-600 hover:underline">← Back to Home</a>
                </div>
                
                <div class="mb-6">
                    <h2 class="text-lg font-semibold mb-3 text-green-700">📊 Combined Documents ({len(combined_files)})</h2>
                    <div class="grid grid-cols-2 gap-3">
                        {''.join(f'''
                        <a href="/view-file/{folder_name}/{f}" 
                           class="p-3 bg-green-50 border border-green-200 rounded-lg hover:bg-green-100 transition">
                            <div class="font-medium text-green-800">{f.replace('combined_', '').replace('.json', '').replace('_', ' ').title()}</div>
                            <div class="text-xs text-green-600">{f}</div>
                        </a>
                        ''' for f in combined_files)}
                    </div>
                </div>
                
                <div class="mb-6">
                    <h2 class="text-lg font-semibold mb-3 text-blue-700">📄 Page Extractions ({len(page_files)})</h2>
                    <div class="flex flex-wrap gap-2">
                        {''.join(f'''
                        <a href="/view-file/{folder_name}/{f}" 
                           class="px-3 py-2 bg-blue-50 border border-blue-200 rounded hover:bg-blue-100 text-sm">
                            {f.replace('.json', '').replace('page_', 'Page ')}
                        </a>
                        ''' for f in page_files)}
                    </div>
                </div>
                
                {f'''
                <div>
                    <h2 class="text-lg font-semibold mb-3 text-gray-700">📁 Other Files ({len(other_files)})</h2>
                    <div class="flex flex-wrap gap-2">
                        {''.join(f"""
                        <a href="/view-file/{folder_name}/{f}" 
                           class="px-3 py-2 bg-gray-50 border border-gray-200 rounded hover:bg-gray-100 text-sm">
                            {f}
                        </a>
                        """ for f in other_files)}
                    </div>
                </div>
                ''' if other_files else ''}
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.get("/api/page-extraction/{folder_name}")
async def get_page_extraction(folder_name: str, _=Depends(require_login)):
    """Get details of a specific page extraction."""
    from pathlib import Path
    folder = Path("processed") / folder_name
    
    if not folder.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    
    result_file = folder / "extraction_result.json"
    if not result_file.exists():
        return JSONResponse({"error": "No extraction result"}, status_code=404)
    
    with open(result_file) as f:
        data = json.load(f)
    
    # Include combined document data
    combined = {}
    for combined_file in folder.glob("combined_*.json"):
        with open(combined_file) as f:
            combined[combined_file.stem] = json.load(f)
    
    data["combined_documents"] = combined
    return JSONResponse(data)


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request, db: Session = Depends(get_db), _=Depends(require_login)):
    uploads = db.query(Upload).order_by(Upload.created_at.desc()).all()
    return templates.TemplateResponse("history.html", {"request": request, "uploads": uploads, "auth_enabled": AUTH_ENABLED})


@app.get("/view/{upload_id}", response_class=HTMLResponse)
async def view_json(upload_id: str, request: Request, db: Session = Depends(get_db), _=Depends(require_login)):
    upload = db.query(Upload).filter(Upload.id == upload_id).first()
    if not upload:
        return HTMLResponse("Not found", status_code=404)

    with open(upload.json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    return templates.TemplateResponse(
        "view_json.html",
        {
            "request": request,
            "upload": upload,
            "json_data": json_data,
            "auth_enabled": AUTH_ENABLED,
        },
    )


# ---- Auth routes ----

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if not AUTH_ENABLED:
        # If auth disabled, redirect to home
        return RedirectResponse("/")
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    success = await login_user(request, username, password)
    if not success:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid credentials"},
            status_code=401,
        )
    return RedirectResponse("/", status_code=302)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/")
