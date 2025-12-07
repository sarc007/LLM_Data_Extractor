import os
import uuid
import json
import shutil

from fastapi import FastAPI, UploadFile, Form, Depends, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

load_dotenv()

from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse

from .database import Base, engine, get_db
from .models import Upload
from .pipeline import process_document
from .auth import require_login, login_user, AUTH_ENABLED

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

    # Run pipeline
    try:
        json_result, model_used = process_document(upload_path, "format_template.json")
    except Exception as e:
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
    return FileResponse(
        upload.json_path,
        media_type="application/json",
        filename=f"{upload.id}.json"
    )


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
