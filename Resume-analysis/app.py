"""
Resume Analysis API — FastAPI Backend

Endpoints:
  POST /analyze      — Analyze resume text
  POST /analyze/pdf  — Upload & analyze PDF resume
  GET  /health       — Health check + model status
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.schemas import AnalyzeRequest, AnalyzeResponse, ResumeAnalysis
from core.model_loader import load_model, get_model_info, is_loaded
from core.analyzer import analyze_resume
from core.pdf_parser import extract_text_from_pdf

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan — load model on startup ──────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting model loading...")
    try:
        load_model()
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        logger.warning("API will start but /analyze endpoints will fail.")
    yield
    logger.info("Shutting down.")


# ── App ────────────────────────────────────────────────────
app = FastAPI(
    title="Resume Analysis API",
    description="AI-powered resume analysis using fine-tuned Qwen2 with LoRA adapter",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ───────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check with model status."""
    info = get_model_info()
    return {
        "status": "healthy",
        "model_loaded": is_loaded(),
        "model_info": info,
    }


# ── Analyze Text ──────────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze resume from plain text.

    Accepts resume text and optional job description.
    Returns structured analysis with 10 evaluation fields.
    """
    if not is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for initialization.",
        )

    start = time.time()

    try:
        analysis = analyze_resume(
            resume_text=request.resume_text,
            job_description=request.job_description,
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    elapsed = time.time() - start
    logger.info(f"Analysis completed in {elapsed:.2f}s")

    return AnalyzeResponse(
        success=True,
        analysis=analysis,
        model_info=f"{get_model_info()['base_model']} + {get_model_info()['adapter']}",
        processing_time_seconds=round(elapsed, 2),
    )


# ── Analyze PDF ───────────────────────────────────────────
@app.post("/analyze/pdf", response_model=AnalyzeResponse)
async def analyze_pdf(
    file: UploadFile = File(...),
    job_description: str = Form(default=""),
):
    """
    Analyze resume from uploaded PDF.

    Extracts text from PDF, then performs structured analysis.
    """
    if not is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for initialization.",
        )

    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Read PDF
    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    if len(pdf_bytes) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=400, detail="File too large. Max 10 MB.")

    # Extract text
    resume_text = extract_text_from_pdf(pdf_bytes)
    if not resume_text or len(resume_text.strip()) < 50:
        raise HTTPException(
            status_code=422,
            detail="Could not extract sufficient text from PDF. Please ensure it's not scanned/image-based.",
        )

    start = time.time()

    try:
        analysis = analyze_resume(
            resume_text=resume_text,
            job_description=job_description,
        )
    except Exception as e:
        logger.error(f"PDF analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    elapsed = time.time() - start
    logger.info(f"PDF analysis completed in {elapsed:.2f}s")

    return AnalyzeResponse(
        success=True,
        analysis=analysis,
        model_info=f"{get_model_info()['base_model']} + {get_model_info()['adapter']}",
        processing_time_seconds=round(elapsed, 2),
    )


# ── Run ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
